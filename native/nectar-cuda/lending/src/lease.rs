//! The lender's lease: one borrower at a time.
//!
//! A borrower takes the lease with its control connection, proving the shared secret, and
//! gets a token its CUDA connections prove in their turn. The lease outlives a short drop:
//! it is released only once the holder has no connection left and has not been heard from
//! (a heartbeat, a connection opened or closed) for GRACE. A borrower that comes back
//! within that time gets the same lease and token back, so programs that kept running
//! carry on. GRACE is longer than the borrower's own grace (nectar-svc), so a borrower
//! that is still trying always finds its lease.
//!
//! Times are passed in, so tests run on a made-up clock.

use std::time::{Duration, Instant};

pub const GRACE: Duration = Duration::from_secs(30);

#[derive(Default)]
pub struct Lease {
    holder: Option<Holder>,
}

struct Holder {
    name: String,
    token: String,
    /// Open control and CUDA connections
    connections: u32,
    last_heard: Instant,
}

impl Lease {
    /// The holder's name, after releasing a lease whose grace has run out
    pub fn holder(&mut self, now: Instant) -> Option<&str> {
        self.expire(now);
        self.holder.as_ref().map(|holder| holder.name.as_str())
    }

    /// Takes the lease for a control connection, or takes it back within its grace. Returns
    /// the token, or the name of the borrower that holds it.
    pub fn take(
        &mut self,
        name: &str,
        now: Instant,
        new_token: impl FnOnce() -> String,
    ) -> Result<String, String> {
        self.expire(now);
        match &mut self.holder {
            Some(holder) if holder.name == name => {
                holder.connections += 1;
                holder.last_heard = now;
                Ok(holder.token.clone())
            }
            Some(holder) => Err(holder.name.clone()),
            None => {
                let token = new_token();
                self.holder = Some(Holder {
                    name: name.to_string(),
                    token: token.clone(),
                    connections: 1,
                    last_heard: now,
                });
                Ok(token)
            }
        }
    }

    /// The token CUDA connections must prove, while the lease is held
    pub fn token(&mut self, now: Instant) -> Option<String> {
        self.expire(now);
        self.holder.as_ref().map(|holder| holder.token.clone())
    }

    /// A CUDA connection proving `token`: whether the lease is its
    pub fn open(&mut self, token: &str, now: Instant) -> bool {
        self.expire(now);
        match self.holding(token) {
            Some(holder) => {
                holder.connections += 1;
                holder.last_heard = now;
                true
            }
            None => false,
        }
    }

    /// A connection that `open` or `take` counted has closed
    pub fn close(&mut self, token: &str, now: Instant) {
        if let Some(holder) = self.holding(token) {
            holder.connections = holder.connections.saturating_sub(1);
            holder.last_heard = now;
        }
    }

    /// A heartbeat on the control connection: whether the lease is still its
    pub fn beat(&mut self, token: &str, now: Instant) -> bool {
        match self.holding(token) {
            Some(holder) => {
                holder.last_heard = now;
                true
            }
            None => false,
        }
    }

    /// The borrower gives the lease back
    pub fn give_back(&mut self, token: &str) {
        if self.holding(token).is_some() {
            self.holder = None;
        }
    }

    fn holding(&mut self, token: &str) -> Option<&mut Holder> {
        self.holder
            .as_mut()
            .filter(|holder| crate::crypto::same(holder.token.as_bytes(), token.as_bytes()))
    }

    fn expire(&mut self, now: Instant) {
        if let Some(holder) = &self.holder
            && holder.connections == 0
            && now.saturating_duration_since(holder.last_heard) >= GRACE
        {
            self.holder = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECOND: Duration = Duration::from_secs(1);

    fn token() -> String {
        "t1".to_string()
    }

    #[test]
    fn one_borrower_at_a_time() {
        let start = Instant::now();
        let mut lease = Lease::default();
        assert_eq!(lease.take("laptop", start, token), Ok("t1".into()));
        assert_eq!(lease.take("other", start, || "t2".into()), Err("laptop".into()));
        assert!(lease.open("t1", start));
        assert!(!lease.open("t2", start));
        assert_eq!(lease.holder(start), Some("laptop"));

        lease.give_back("t1");
        assert_eq!(lease.holder(start), None);
        assert_eq!(lease.take("other", start, || "t2".into()), Ok("t2".into()));
    }

    #[test]
    fn grace_with_live_connections_never_ends() {
        let start = Instant::now();
        let mut lease = Lease::default();
        lease.take("laptop", start, token).unwrap();
        assert!(lease.open("t1", start));
        // The control connection is gone, a CUDA connection stays: no heartbeats for long
        lease.close("t1", start);
        assert_eq!(lease.holder(start + GRACE * 10), Some("laptop"));
        assert_eq!(lease.take("other", start + GRACE * 10, || "t2".into()), Err("laptop".into()));
    }

    #[test]
    fn grace_with_dead_connections_ends_after_grace() {
        let start = Instant::now();
        let mut lease = Lease::default();
        lease.take("laptop", start, token).unwrap();
        lease.beat("t1", start + 5 * SECOND);
        lease.close("t1", start + 6 * SECOND);
        // Still within grace of the last time it was heard from
        assert_eq!(lease.holder(start + 6 * SECOND + GRACE - SECOND), Some("laptop"));
        assert_eq!(lease.holder(start + 6 * SECOND + GRACE), None);
        assert!(!lease.open("t1", start + 6 * SECOND + GRACE));
    }

    #[test]
    fn coming_back_within_grace_keeps_the_token() {
        let start = Instant::now();
        let mut lease = Lease::default();
        lease.take("laptop", start, token).unwrap();
        lease.close("t1", start + SECOND);
        let back = start + SECOND + GRACE - SECOND;
        assert_eq!(lease.take("laptop", back, || "t2".into()), Ok("t1".into()));
        assert!(lease.beat("t1", back + SECOND));
        // And it is counted again: no expiry while its control connection is open
        assert_eq!(lease.holder(back + GRACE * 3), Some("laptop"));
    }

    #[test]
    fn coming_back_after_grace_gets_a_new_lease() {
        let start = Instant::now();
        let mut lease = Lease::default();
        lease.take("laptop", start, token).unwrap();
        lease.close("t1", start);
        assert_eq!(lease.take("laptop", start + GRACE, || "t2".into()), Ok("t2".into()));
        assert!(!lease.open("t1", start + GRACE));
    }
}
