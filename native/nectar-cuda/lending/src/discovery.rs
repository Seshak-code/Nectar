//! Finding lenders. A borrower broadcasts a probe on UDP port 43211; each lender that knows
//! the same secret answers to the probe's sender, so the borrower needs no firewall rule.
//!
//! Both are text, one key=value per line after the first:
//!
//! ```text
//! NECTAR-LEND 1 probe          NECTAR-LEND 1 lender
//! nonce=<32 hex digits>        nonce=<the probe's nonce>
//! time=<unix seconds>          name=DESKTOP  gpu=…  vram=…  driver=…  port=…  state=…
//!                              borrower=…  hybrid=<hybrid runtime key>
//! mac=<HMAC of the lines above under the secret>
//! ```
//!
//! A lender ignores probes without a valid mac, or whose time is more than MAX_SKEW from its
//! own. The answer repeats the probe's nonce under its mac, so a borrower accepts only fresh
//! answers from lenders that know the secret.

use crate::crypto::{hex, hmac, random, same, unhex};

pub const PORT: u16 = 43211;
const PROBE: &str = "NECTAR-LEND 1 probe";
const ANSWER: &str = "NECTAR-LEND 1 lender";

/// How far a probe's time may be from the lender's clock, in seconds
pub const MAX_SKEW: u64 = 300;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Available,
    Lent,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LenderInfo {
    /// The lender's computer name
    pub name: String,
    pub gpu: String,
    pub vram_mib: u64,
    /// The CUDA version its driver supports, such as 13.5
    pub driver: String,
    pub port: u16,
    pub state: State,
    /// Who has the lease, when lent
    pub borrower: String,
    /// The key of its driver's hybrid runtime (crate::hybrid_key), so a borrower fetches
    /// the file only when its sandbox lacks it. Empty when the lender has none.
    pub hybrid: String,
}

/// A probe's nonce, kept by the borrower to check answers against
pub type Nonce = [u8; 16];

/// A new probe and its nonce
pub fn probe(secret: &[u8], now: u64) -> (String, Nonce) {
    let nonce = random::<16>();
    let body = format!("{PROBE}\nnonce={}\ntime={now}\n", hex(&nonce));
    (sign(secret, body), nonce)
}

/// The nonce of a valid, fresh probe
pub fn check_probe(secret: &[u8], text: &str, now: u64) -> Option<Nonce> {
    let fields = verify(secret, text, PROBE)?;
    let time: u64 = field(&fields, "time")?.parse().ok()?;
    if time.abs_diff(now) > MAX_SKEW {
        return None;
    }
    unhex(field(&fields, "nonce")?)?.try_into().ok()
}

/// The answer to a probe
pub fn answer(secret: &[u8], nonce: &Nonce, info: &LenderInfo) -> String {
    let state = match info.state {
        State::Available => "available",
        State::Lent => "lent",
    };
    let apis: String = info
        .apis
        .iter()
        .map(|offer| format!("api={}\n", clean(&offer.text())))
        .collect();
    let body = format!(
        "{ANSWER}\nnonce={}\nname={}\ngpu={}\nvram={}\nvendor={}\nos={}\narch={}\n{apis}port={}\nstate={state}\nborrower={}\nhybrid={}\n",
        hex(nonce),
        clean(&info.name),
        clean(&info.gpu),
        info.vram_mib,
        clean(&info.vendor),
        clean(&info.os),
        clean(&info.arch),
        info.port,
        clean(&info.borrower),
        clean(&info.hybrid),
    );
    sign(secret, body)
}

/// The lender an answer describes, if it answers this probe and knows the secret
pub fn check_answer(secret: &[u8], text: &str, nonce: &Nonce) -> Option<LenderInfo> {
    let fields = verify(secret, text, ANSWER)?;
    if !same(&unhex(field(&fields, "nonce")?)?, nonce) {
        return None;
    }
    Some(LenderInfo {
        name: field(&fields, "name")?.to_string(),
        gpu: field(&fields, "gpu")?.to_string(),
        vram_mib: field(&fields, "vram")?.parse().ok()?,
        driver: field(&fields, "driver")?.to_string(),
        port: field(&fields, "port")?.parse().ok()?,
        state: match field(&fields, "state")? {
            "lent" => State::Lent,
            _ => State::Available,
        },
        borrower: field(&fields, "borrower")?.to_string(),
        hybrid: field(&fields, "hybrid").unwrap_or_default().to_string(),
    })
}

fn sign(secret: &[u8], body: String) -> String {
    let mac = hex(&hmac(secret, body.as_bytes()));
    format!("{body}mac={mac}\n")
}

/// The fields of a message that starts with `first` and whose mac is right
fn verify<'a>(secret: &[u8], text: &'a str, first: &str) -> Option<Vec<(&'a str, &'a str)>> {
    let at = text.rfind("mac=")?;
    let (body, mac) = text.split_at(at);
    let mac = unhex(mac.trim_start_matches("mac=").trim_end())?;
    if !body.starts_with(&format!("{first}\n")) || !same(&mac, &hmac(secret, body.as_bytes())) {
        return None;
    }
    Some(
        body.lines()
            .skip(1)
            .filter_map(|line| line.split_once('='))
            .collect(),
    )
}

fn field<'a>(fields: &[(&'a str, &'a str)], key: &str) -> Option<&'a str> {
    fields.iter().find(|(k, _)| *k == key).map(|(_, value)| *value)
}

/// A value on one line
fn clean(value: &str) -> String {
    value.replace(['\r', '\n'], " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desktop() -> LenderInfo {
        LenderInfo {
            name: "DESKTOP".into(),
            gpu: "NVIDIA GeForce RTX 5070".into(),
            vram_mib: 12199,
            driver: "13.5".into(),
            port: 43212,
            state: State::Lent,
            borrower: "NRUSIMHA".into(),
            hybrid: "13050-0011aabbccdd".into(),
        }
    }

    #[test]
    fn a_probe_and_its_answer_round_trip() {
        let now = 1_800_000_000;
        let (text, nonce) = probe(b"secret", now);
        assert_eq!(check_probe(b"secret", &text, now + 10), Some(nonce));

        let answer = answer(b"secret", &nonce, &desktop());
        assert_eq!(check_answer(b"secret", &answer, &nonce), Some(desktop()));
    }

    #[test]
    fn a_wrong_secret_gets_nothing() {
        let now = 1_800_000_000;
        let (text, nonce) = probe(b"secret", now);
        assert_eq!(check_probe(b"guess", &text, now), None);

        let answer = answer(b"guess", &nonce, &desktop());
        assert_eq!(check_answer(b"secret", &answer, &nonce), None);
    }

    #[test]
    fn stale_probes_and_answers_to_other_probes_are_ignored() {
        let now = 1_800_000_000;
        let (text, nonce) = probe(b"secret", now);
        assert_eq!(check_probe(b"secret", &text, now + MAX_SKEW + 1), None);

        // A replayed answer to an earlier probe
        let (_, earlier) = probe(b"secret", now);
        let replayed = answer(b"secret", &earlier, &desktop());
        assert_eq!(check_answer(b"secret", &replayed, &nonce), None);
    }

    #[test]
    fn tampered_messages_are_ignored() {
        let now = 1_800_000_000;
        let (_, nonce) = probe(b"secret", now);
        let answer = answer(b"secret", &nonce, &desktop());
        let tampered = answer.replace("state=lent", "state=available");
        assert_eq!(check_answer(b"secret", &tampered, &nonce), None);

        let mut info = desktop();
        info.name = "EVIL\nstate=available".into();
        let answer = super::answer(b"secret", &nonce, &info);
        assert_eq!(check_answer(b"secret", &answer, &nonce).unwrap().state, State::Lent);
    }
}
