//! The model of the link: when each packet arrives.
//!
//! A direction sends one packet at a time, so packets queue behind each other at its rate. A packet
//! then takes half the round trip, plus jitter, and lost ones take longer: a retry at the Wi-Fi
//! layer is short, and now and then one exhausts its retries and waits for TCP. Packets arrive in
//! order, as TCP delivers them.
//!
//! Attenuation lowers the signal, which lowers the modulation the radios can hold, which lowers the
//! rate and raises the loss and jitter. The table is the usual shape for 802.11ax at 160 MHz with
//! two streams. It's an estimate: only `measured` at no attenuation comes from real radios.

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// A link at full signal
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Link {
    pub name: &'static str,
    /// The round trip when nothing's queued, in milliseconds
    pub rtt_ms: f64,
    /// How far the one-way delay wanders, in milliseconds
    pub jitter_ms: f64,
    /// Megabytes a second the borrower can send and receive
    pub up_mb_s: f64,
    pub down_mb_s: f64,
    /// The share of packets the Wi-Fi layer has to send again
    pub retry: f64,
    /// The signal to noise ratio, in dB
    pub snr_db: f64,
}

impl Link {
    pub const ALL: [Link; 5] = [
        // What the laptop (Wi-Fi 7) measured against the desktop (Wi-Fi 6E), through the router
        Link { name: "measured", rtt_ms: 6.0, jitter_ms: 2.5, up_mb_s: 70.0, down_mb_s: 41.0, retry: 0.02, snr_db: 38.0 },
        // Estimates from here on
        Link { name: "cabled-lender", rtt_ms: 3.5, jitter_ms: 1.5, up_mb_s: 110.0, down_mb_s: 110.0, retry: 0.01, snr_db: 42.0 },
        Link { name: "wifi7-both", rtt_ms: 4.0, jitter_ms: 1.5, up_mb_s: 180.0, down_mb_s: 180.0, retry: 0.02, snr_db: 40.0 },
        Link { name: "busy", rtt_ms: 12.0, jitter_ms: 8.0, up_mb_s: 25.0, down_mb_s: 20.0, retry: 0.08, snr_db: 30.0 },
        Link { name: "ideal", rtt_ms: 0.0, jitter_ms: 0.0, up_mb_s: 0.0, down_mb_s: 0.0, retry: 0.0, snr_db: 60.0 },
    ];

    pub fn by_name(name: &str) -> Option<Link> {
        Link::ALL.into_iter().find(|link| link.name == name)
    }

    pub fn about(&self) -> String {
        if self.up_mb_s == 0.0 {
            return "no limits, to see what the proxy itself costs".to_string();
        }
        format!(
            "{} ms round trip, +/-{} ms, {} MB/s up, {} MB/s down, {}% sent again",
            self.rtt_ms,
            self.jitter_ms,
            self.up_mb_s,
            self.down_mb_s,
            self.retry * 100.0
        )
    }
}

/// (the SNR a modulation needs in dB, its rate in Mbit/s), best first
const MODULATIONS: [(f64, f64); 12] = [
    (36.0, 1201.0),
    (33.0, 1081.0),
    (31.0, 961.0),
    (28.0, 865.0),
    (25.0, 721.0),
    (21.0, 648.0),
    (19.0, 576.0),
    (16.0, 432.0),
    (13.0, 288.0),
    (10.0, 216.0),
    (7.0, 144.0),
    (4.0, 72.0),
];

/// What a signal of `snr` dB gives: the modulation's rate, and how close it is to the next one down
fn modulation(snr: f64) -> (f64, f64) {
    for (needs, rate) in MODULATIONS {
        if snr >= needs {
            return (rate, snr - needs);
        }
    }
    // Below the lowest: barely connected
    (36.0, -1.0)
}

/// The link as a signal `attenuation` dB weaker leaves it
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct State {
    pub rate_share: f64,
    pub retry: f64,
    pub extra_rtt_ms: f64,
    pub extra_jitter_ms: f64,
}

pub fn state(link: &Link, attenuation: f64) -> State {
    let (full, _) = modulation(link.snr_db);
    let snr = link.snr_db - attenuation;
    let (rate, margin) = modulation(snr);
    // Close above a modulation's limit, frames start failing before the rate falls back
    let edge = if margin < 1.5 { 0.15 * (1.5 - margin) / 1.5 } else { 0.0 };
    // Each 10 dB lost costs airtime in retries and in rate control's searching
    let weakness = (attenuation / 10.0).max(0.0);
    State {
        rate_share: (rate / full).min(1.0),
        retry: (link.retry + edge + 0.01 * weakness).min(0.6),
        extra_rtt_ms: 0.5 * weakness * weakness,
        extra_jitter_ms: 0.8 * weakness * weakness,
    }
}

/// xorshift: the same run behaves the same way
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// When a direction is next free, and the last arrival so nothing overtakes it
struct Direction {
    free_at: Instant,
    last: Instant,
}

/// A link in use, shared by every connection over it
pub struct Air {
    link: Link,
    attenuation: f64,
    /// Seconds per swing, and how many dB it swings by
    fade: Option<(f64, f64)>,
    started: Instant,
    up: Mutex<Direction>,
    down: Mutex<Direction>,
}

impl Air {
    pub fn new(link: Link, attenuation: f64, fade: Option<(f64, f64)>) -> Self {
        let now = Instant::now();
        let direction = || Mutex::new(Direction { free_at: now, last: now });
        Self { link, attenuation, fade, started: now, up: direction(), down: direction() }
    }

    pub fn describe(&self) -> String {
        let mut text = format!("{}: {}", self.link.name, self.link.about());
        if self.attenuation != 0.0 {
            text += &format!(", {} dB weaker", self.attenuation);
        }
        if let Some((period, depth)) = self.fade {
            text += &format!(", fading by up to {depth} dB every {period} s");
        }
        text
    }

    /// The attenuation at `now`, with the fade: smooth, then a dip
    fn attenuation_at(&self, now: Instant) -> f64 {
        let Some((period, depth)) = self.fade else { return self.attenuation };
        let phase = (now - self.started).as_secs_f64() / period.max(0.1) * std::f64::consts::TAU;
        self.attenuation + depth * (0.5 - 0.5 * phase.cos()).powi(2)
    }

    /// When a packet of `bytes` handed over at `now` arrives
    pub fn schedule(&self, up: bool, bytes: usize, now: Instant, rng: &mut Rng) -> Instant {
        let link = &self.link;
        let rate = if up { link.up_mb_s } else { link.down_mb_s };
        if rate == 0.0 {
            return now;
        }
        let state = state(link, self.attenuation_at(now));
        let mut direction = (if up { &self.up } else { &self.down }).lock().unwrap_or_else(|p| p.into_inner());
        let mut airtime = bytes as f64 / (rate * state.rate_share * 1e6);
        let mut extra = 0.0;
        if rng.next() < state.retry {
            // Sent again: the air again, and a little backoff
            airtime *= 2.0;
            extra += 0.0005 + rng.next() * 0.0015;
            // One in fifty exhausts its retries, and TCP notices a round trip or two later
            if rng.next() < 0.02 {
                extra += (link.rtt_ms + state.extra_rtt_ms) * 1e-3 * (1.0 + rng.next());
            }
        }
        let start = direction.free_at.max(now);
        direction.free_at = start + Duration::from_secs_f64(airtime);
        let one_way = (link.rtt_ms + state.extra_rtt_ms) * 0.5e-3;
        let jitter = (link.jitter_ms + state.extra_jitter_ms) * 1e-3 * rng.next();
        let due = direction.free_at + Duration::from_secs_f64(one_way + jitter + extra);
        direction.last = direction.last.max(due);
        direction.last
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng() -> Rng {
        Rng::new(42)
    }

    #[test]
    fn every_profile_has_a_name_and_a_description() {
        for link in Link::ALL {
            assert_eq!(Link::by_name(link.name), Some(link));
            assert!(!link.about().is_empty());
        }
        assert_eq!(Link::by_name("carrier pigeon"), None);
    }

    #[test]
    fn full_signal_changes_nothing_much() {
        let link = Link::by_name("measured").unwrap();
        let full = state(&link, 0.0);
        assert_eq!(full.rate_share, 1.0);
        assert_eq!(full.extra_rtt_ms, 0.0);
    }

    #[test]
    fn a_weaker_signal_is_never_better() {
        let link = Link::by_name("measured").unwrap();
        let mut last = state(&link, 0.0);
        for db in 1..35 {
            let now = state(&link, db as f64);
            assert!(now.rate_share <= last.rate_share, "{db} dB");
            assert!(now.extra_rtt_ms >= last.extra_rtt_ms, "{db} dB");
            last = now;
        }
        assert!(state(&link, 20.0).rate_share < 0.6);
        assert!(state(&link, 34.0).rate_share < 0.2);
    }

    #[test]
    fn rate_limits_what_crosses() {
        let air = Air::new(Link { retry: 0.0, jitter_ms: 0.0, ..Link::by_name("measured").unwrap() }, 0.0, None);
        let start = Instant::now();
        let mut random = rng();
        let mut last = start;
        // 7 MB up at 70 MB/s
        for _ in 0..(7_000_000 / 16384) {
            last = air.schedule(true, 16384, start, &mut random);
        }
        let seconds = (last - start).as_secs_f64();
        assert!((0.098..0.105).contains(&seconds), "{seconds}");
        // Down is free: another direction, and slower
        let down = air.schedule(false, 41_000_000, start, &mut random);
        assert!(((down - start).as_secs_f64() - 1.003).abs() < 0.002);
    }

    #[test]
    fn packets_arrive_in_order() {
        let air = Air::new(Link::by_name("busy").unwrap(), 12.0, Some((4.0, 10.0)));
        let start = Instant::now();
        let mut random = rng();
        let mut last = start;
        for index in 0..5000 {
            let due = air.schedule(index % 2 == 0, 4096, start + Duration::from_micros(index), &mut random);
            if index % 2 == 0 {
                assert!(due >= last);
                last = due;
            }
        }
    }

    #[test]
    fn fades_come_and_go() {
        let air = Air::new(Link::by_name("measured").unwrap(), 5.0, Some((8.0, 10.0)));
        assert!((air.attenuation_at(air.started) - 5.0).abs() < 1e-9);
        assert!((air.attenuation_at(air.started + Duration::from_secs(4)) - 15.0).abs() < 1e-9);
    }

    #[test]
    fn ideal_adds_nothing() {
        let air = Air::new(Link::by_name("ideal").unwrap(), 0.0, None);
        let now = Instant::now();
        assert_eq!(air.schedule(true, 1 << 20, now, &mut rng()), now);
    }
}
