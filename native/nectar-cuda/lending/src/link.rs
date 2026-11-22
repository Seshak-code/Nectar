//! The link check: measures the connection to the lender as it is tuned for the chosen profile,
//! both ends say what they're connected by, and the result turns into advice for that profile.
//! It runs over a connection of its own (protocol::HELLO_LINK), proving the lease's token.

use std::io::{self, BufReader, BufWriter, Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

use nectar_cuda_protocol::{self as protocol, LINK_ECHO, LINK_INFO, LINK_SINK, LINK_SINK_END, LINK_SOURCE, Reader, Writer};

use crate::profile::Profile;
use crate::wifi::{self, Kind};

const ECHOES: usize = 100;
const ECHO_EVERY: Duration = Duration::from_millis(10);
/// How long each direction is measured, and in what pieces
const MEASURE_FOR: Duration = Duration::from_secs(2);
const CHUNK: usize = 1 << 20;
/// Pieces asked for at once on the way down
const SOURCE_CHUNKS: u64 = 4;

/// One end of the connection
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct End {
    pub kind: Kind,
    /// Such as "Wi-Fi 7"
    pub standard: String,
    pub receive_mbps: u32,
    pub transmit_mbps: u32,
    pub signal: u32,
    /// Bytes this end has sent over again on this connection, where the system says
    pub resent: Option<u64>,
}

impl End {
    /// This computer's end of `stream`
    fn here(stream: &TcpStream) -> End {
        let mut end = End {
            kind: stream.local_addr().map_or(Kind::Unknown, |local| wifi::kind_of(local.ip())),
            resent: resent(stream),
            ..End::default()
        };
        if end.kind == Kind::Wifi
            && let Some(link) = wifi::links().into_iter().next()
        {
            end.standard = link.standard;
            end.receive_mbps = link.receive_mbps;
            end.transmit_mbps = link.transmit_mbps;
            end.signal = link.signal;
        }
        end
    }

    fn encode(&self, w: &mut Writer) {
        w.bytes(self.kind.key().as_bytes())
            .bytes(self.standard.as_bytes())
            .u32(self.receive_mbps)
            .u32(self.transmit_mbps)
            .u32(self.signal)
            .u8(self.resent.is_some() as u8)
            .u64(self.resent.unwrap_or(0));
    }

    fn decode(r: &mut Reader) -> io::Result<End> {
        let kind = Kind::parse(&String::from_utf8_lossy(r.bytes()?));
        let standard = String::from_utf8_lossy(r.bytes()?).into_owned();
        let (receive_mbps, transmit_mbps, signal) = (r.u32()?, r.u32()?, r.u32()?);
        let known = r.u8()? != 0;
        let resent = r.u64()?;
        Ok(End { kind, standard, receive_mbps, transmit_mbps, signal, resent: known.then_some(resent) })
    }

    /// As the report shows it
    fn describe(&self) -> String {
        match self.kind {
            Kind::Wifi if self.signal == 0 => "Wi-Fi".to_string(),
            Kind::Wifi if self.standard.is_empty() => format!("Wi-Fi, signal {}%", self.signal),
            Kind::Wifi => format!(
                "{}, {} Mbit/s up and {} down, signal {}%",
                self.standard, self.transmit_mbps, self.receive_mbps, self.signal
            ),
            Kind::Wired => "a cable".to_string(),
            Kind::Loopback => "loopback".to_string(),
            Kind::Unknown => "a network Nectar can't identify".to_string(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    pub profile: Profile,
    /// Round trips in milliseconds: the median, the worst but one in twenty, and the worst
    pub rtt_ms: f64,
    pub rtt_p95_ms: f64,
    pub rtt_max_ms: f64,
    pub up_mb_s: f64,
    pub down_mb_s: f64,
    /// The share of what crossed that had to be sent again, in percent, where known
    pub up_resent: Option<f64>,
    pub down_resent: Option<f64>,
    pub here: End,
    pub lender: End,
}

fn percent(resent: Option<u64>, before: Option<u64>, bytes: u64) -> Option<f64> {
    Some(resent?.saturating_sub(before?) as f64 * 100.0 / bytes.max(1) as f64)
}

fn percentile(sorted: &[f64], share: f64) -> f64 {
    sorted.get(((sorted.len() as f64 * share) as usize).min(sorted.len().saturating_sub(1))).copied().unwrap_or(0.0)
}

fn lender_end(reader: &mut impl Read, writer: &mut impl Write) -> io::Result<End> {
    protocol::write_message(writer, LINK_INFO, &[])?;
    writer.flush()?;
    let (_, body) = protocol::read_message(reader)?;
    End::decode(&mut Reader::new(&body))
}

/// Measures the lender at `server`
pub fn check(server: &str, token: &str, profile: Profile) -> io::Result<Report> {
    let stream = crate::connect(server, token, protocol::HELLO_LINK, Duration::from_secs(2))?;
    stream.set_read_timeout(Some(Duration::from_secs(15)))?;
    let mut reader = BufReader::with_capacity(1 << 16, stream.try_clone()?);
    let mut writer = BufWriter::with_capacity(1 << 16, stream.try_clone()?);

    let mut trips = Vec::with_capacity(ECHOES);
    for _ in 0..ECHOES {
        let sent = Instant::now();
        protocol::write_message(&mut writer, LINK_ECHO, &[])?;
        writer.flush()?;
        protocol::read_message(&mut reader)?;
        let trip = sent.elapsed();
        trips.push(trip.as_secs_f64() * 1000.0);
        std::thread::sleep(ECHO_EVERY.saturating_sub(trip));
    }
    trips.sort_by(f64::total_cmp);

    // Up: until the lender says it has it all, so nothing is counted that only reached a buffer
    let before = End::here(&stream).resent;
    let chunk = vec![0x5au8; CHUNK];
    let started = Instant::now();
    let mut sent = 0u64;
    while started.elapsed() < MEASURE_FOR {
        protocol::write_message(&mut writer, LINK_SINK, &[&chunk])?;
        sent += CHUNK as u64;
    }
    protocol::write_message(&mut writer, LINK_SINK_END, &[])?;
    writer.flush()?;
    protocol::read_message(&mut reader)?;
    let up_mb_s = sent as f64 / 1e6 / started.elapsed().as_secs_f64();
    let up_resent = percent(End::here(&stream).resent, before, sent);

    // Down
    let before = lender_end(&mut reader, &mut writer)?.resent;
    let started = Instant::now();
    let mut received = 0u64;
    while started.elapsed() < MEASURE_FOR {
        let mut w = Writer::new();
        w.u64(SOURCE_CHUNKS);
        protocol::write_message(&mut writer, LINK_SOURCE, &[w.as_slice()])?;
        writer.flush()?;
        for _ in 0..SOURCE_CHUNKS {
            received += protocol::read_message(&mut reader)?.1.len() as u64;
        }
    }
    let down_mb_s = received as f64 / 1e6 / started.elapsed().as_secs_f64();
    let lender = lender_end(&mut reader, &mut writer)?;
    let down_resent = percent(lender.resent, before, received);

    Ok(Report {
        profile,
        rtt_ms: percentile(&trips, 0.5),
        rtt_p95_ms: percentile(&trips, 0.95),
        rtt_max_ms: trips.last().copied().unwrap_or(0.0),
        up_mb_s,
        down_mb_s,
        up_resent,
        down_resent,
        here: End::here(&stream),
        lender,
    })
}

/// The lender's side of a link check, until the borrower closes it
pub fn serve(reader: &mut impl Read, writer: &mut impl Write, stream: &TcpStream) -> io::Result<()> {
    let chunk = vec![0xa5u8; CHUNK];
    loop {
        let (request, body) = match protocol::read_message(reader) {
            Ok(message) => message,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::UnexpectedEof | io::ErrorKind::ConnectionReset | io::ErrorKind::ConnectionAborted
                ) =>
            {
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        match request {
            LINK_ECHO | LINK_SINK_END => protocol::write_message(writer, 0, &[])?,
            LINK_SINK => continue,
            LINK_SOURCE => {
                let chunks = Reader::new(&body).u64()?.min(1024);
                for _ in 0..chunks {
                    protocol::write_message(writer, 0, &[&chunk])?;
                }
            }
            LINK_INFO => {
                let mut w = Writer::new();
                End::here(stream).encode(&mut w);
                protocol::write_message(writer, 0, &[w.as_slice()])?;
            }
            _ => protocol::write_message(writer, u32::MAX, &[])?,
        }
        writer.flush()?;
    }
}

impl Report {
    /// What was measured, for the app and the log
    pub fn lines(&self) -> Vec<String> {
        let resent = |share: Option<f64>| share.map_or(String::new(), |share| format!(", {share:.1}% sent twice"));
        vec![
            format!(
                "Round trip {:.1} ms typically, {:.1} ms at 1 in 20, {:.1} ms at worst",
                self.rtt_ms, self.rtt_p95_ms, self.rtt_max_ms
            ),
            format!("Up {:.0} MB/s{}", self.up_mb_s, resent(self.up_resent)),
            format!("Down {:.0} MB/s{}", self.down_mb_s, resent(self.down_resent)),
            format!("This computer is on {}", self.here.describe()),
            format!("The lender is on {}", self.lender.describe()),
        ]
    }

    /// What to do about it, for the profile chosen, most useful first
    pub fn advice(&self) -> Vec<String> {
        let mut advice = Vec::new();
        match (self.here.kind, self.lender.kind) {
            (Kind::Loopback, _) | (_, Kind::Loopback) => {}
            (Kind::Wifi, Kind::Wifi) => advice.push(
                "Both computers are on Wi-Fi, so everything crosses the air twice. A cable on the lender helps most.".to_string(),
            ),
            (_, Kind::Wifi) => advice.push("The lender is on Wi-Fi. A cable on the lender would help most.".to_string()),
            _ => {}
        }
        for (name, end) in [("This computer", &self.here), ("The lender", &self.lender)] {
            if end.kind == Kind::Wifi && end.signal > 0 && end.signal < 60 {
                advice.push(format!("{name}'s Wi-Fi signal is {}%. Moving it closer to the router would help.", end.signal));
            }
            if matches!(end.standard.as_str(), "Wi-Fi 4" | "Wi-Fi 5") {
                advice.push(format!("{name} is on {}. A faster band or card would help.", end.standard));
            }
        }
        let resent = self.up_resent.unwrap_or(0.0).max(self.down_resent.unwrap_or(0.0));
        if resent >= 2.0 {
            advice.push(format!("{resent:.1}% of the data was sent twice. The channel may be busy; try a less crowded one."));
        }
        if self.rtt_p95_ms > 15.0 && self.rtt_p95_ms > self.rtt_ms * 2.0 {
            advice.push(format!(
                "Round trips jump from {:.0} to {:.0} ms. Other traffic on this network may be the cause.",
                self.rtt_ms, self.rtt_p95_ms
            ));
        }
        match self.profile {
            Profile::Ai => {
                let logits_ms = CHUNK as f64 / 1e6 / self.down_mb_s.max(0.001) * 1000.0;
                advice.push(format!(
                    "For AI: each generated token spends about {:.0} ms on this link: a round trip, and {:.0} ms for its 1 MiB of results.",
                    self.rtt_ms + logits_ms,
                    logits_ms
                ));
                advice.push(format!(
                    "For AI: a 6 GB model's first load takes about {:.0} s. CUDA programs' next loads come from the lender's copy.",
                    6000.0 / self.up_mb_s.max(0.001)
                ));
            }
            Profile::Gaming => {
                advice.push(format!(
                    "For games: at 60 fps there's room for about {:.1} MB per frame of commands and changed data.",
                    self.up_mb_s / 60.0
                ));
                if self.rtt_ms > 16.0 {
                    advice.push(format!("For games: a {:.0} ms round trip is longer than a 60 fps frame.", self.rtt_ms));
                }
            }
        }
        advice
    }
}

/// Bytes the system has sent over again on this connection, where it says
#[cfg(windows)]
fn resent(stream: &TcpStream) -> Option<u64> {
    use std::os::windows::io::AsRawSocket;
    use windows_sys::Win32::Networking::WinSock::{SIO_TCP_INFO, TCP_INFO_v0, WSAIoctl};

    let version = 0u32;
    let mut info: TCP_INFO_v0 = unsafe { std::mem::zeroed() };
    let mut returned = 0u32;
    let failed = unsafe {
        WSAIoctl(
            stream.as_raw_socket() as usize,
            SIO_TCP_INFO,
            (&version as *const u32).cast(),
            4,
            (&mut info as *mut TCP_INFO_v0).cast(),
            size_of::<TCP_INFO_v0>() as u32,
            &mut returned,
            std::ptr::null_mut(),
            None,
        )
    };
    (failed == 0).then_some(info.BytesRetrans as u64)
}

#[cfg(target_os = "linux")]
fn resent(stream: &TcpStream) -> Option<u64> {
    use std::os::fd::AsRawFd;

    let mut info: libc::tcp_info = unsafe { std::mem::zeroed() };
    let mut size = size_of::<libc::tcp_info>() as libc::socklen_t;
    let failed = unsafe {
        libc::getsockopt(stream.as_raw_fd(), libc::IPPROTO_TCP, libc::TCP_INFO, (&mut info as *mut libc::tcp_info).cast(), &mut size)
    };
    (failed == 0).then_some(info.tcpi_total_retrans as u64 * info.tcpi_snd_mss as u64)
}

#[cfg(not(any(windows, target_os = "linux")))]
fn resent(_stream: &TcpStream) -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wifi_end(standard: &str, signal: u32) -> End {
        End { kind: Kind::Wifi, standard: standard.to_string(), receive_mbps: 1200, transmit_mbps: 1200, signal, resent: Some(0) }
    }

    fn report(profile: Profile) -> Report {
        Report {
            profile,
            rtt_ms: 6.0,
            rtt_p95_ms: 9.0,
            rtt_max_ms: 25.0,
            up_mb_s: 70.0,
            down_mb_s: 41.0,
            up_resent: Some(0.1),
            down_resent: Some(0.1),
            here: wifi_end("Wi-Fi 7", 93),
            lender: wifi_end("Wi-Fi 6", 80),
        }
    }

    #[test]
    fn ends_cross_the_wire() {
        let end = wifi_end("Wi-Fi 7", 93);
        let mut w = Writer::new();
        end.encode(&mut w);
        assert_eq!(End::decode(&mut Reader::new(w.as_slice())).unwrap(), end);
    }

    #[test]
    fn advice_follows_the_profile() {
        let ai = report(Profile::Ai).advice();
        assert!(ai[0].contains("twice"), "{ai:?}");
        // The measured 30 ms: 6 ms round trip and 26 ms of results at 41 MB/s
        assert!(ai.iter().any(|line| line.contains("about 32 ms")), "{ai:?}");
        let gaming = report(Profile::Gaming).advice();
        assert!(gaming.iter().any(|line| line.contains("1.2 MB per frame")), "{gaming:?}");
        assert!(!gaming.iter().any(|line| line.contains("token")));
    }

    #[test]
    fn advice_names_a_weak_link() {
        let mut weak = report(Profile::Gaming);
        weak.lender = wifi_end("Wi-Fi 5", 40);
        weak.up_resent = Some(3.0);
        weak.rtt_ms = 20.0;
        weak.rtt_p95_ms = 60.0;
        let advice = weak.advice().join("\n");
        for expected in ["signal is 40%", "on Wi-Fi 5", "3.0% of the data", "jump from 20 to 60", "longer than a 60 fps"] {
            assert!(advice.contains(expected), "{expected} not in {advice}");
        }
    }

    #[test]
    fn a_cabled_lender_gets_no_cable_advice() {
        let mut wired = report(Profile::Ai);
        wired.lender = End { kind: Kind::Wired, ..End::default() };
        assert!(!wired.advice().join("\n").contains("cable"));
        assert!(wired.lines().iter().any(|line| line == "The lender is on a cable"));
    }

    #[test]
    fn a_check_runs_against_a_lender() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let server = listener.local_addr().unwrap().to_string();
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut writer = BufWriter::new(stream.try_clone().unwrap());
            assert_eq!(protocol::read_hello(&mut reader).unwrap(), protocol::HELLO_LINK);
            protocol::write_lender_hello(&mut writer, 0, &[]).unwrap();
            serve(&mut reader, &mut writer, &stream).unwrap();
        });
        let report = check(&server, "", Profile::Gaming).unwrap();
        assert!(report.rtt_ms > 0.0 && report.up_mb_s > 0.0 && report.down_mb_s > 0.0);
        assert_eq!(report.here.kind, Kind::Loopback);
        assert_eq!(report.lender.kind, Kind::Loopback);
    }
}
