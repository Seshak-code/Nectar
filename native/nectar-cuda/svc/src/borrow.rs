//! The borrower's side of a lease, as a state machine. Times are passed in, so tests run on
//! a made-up clock.
//!
//! ```text
//! idle ── borrow ──▶ connecting ──▶ borrowed ◀──────────────┐
//!                       │              │ heartbeat missed   │ back within GRACE
//!                       ▼              ▼                    │
//!                     no host ◀── reconnecting ─────────────┘
//!                    (refused)     after GRACE: borrow.cfg removed
//! ```
//!
//! borrow.cfg stays during the grace, so programs that keep running, and new ones, carry
//! on if the lender comes back: the lender keeps the lease (and its token) longer than
//! GRACE. After the grace there is no GPU until the lender answers again.

use std::time::{Duration, Instant};

/// How long a lost lender may take to come back before its GPU is gone
pub const GRACE: Duration = Duration::from_secs(20);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Connecting,
    Borrowed,
    Reconnecting,
    NoHost,
    /// The lender refused, with why: lent to someone else, or the secret differs
    Refused(String),
}

/// What the service does to borrow.cfg
#[derive(Debug, PartialEq, Eq)]
pub enum Action {
    WriteConfig,
    RemoveConfig,
}

pub struct Borrow {
    pub phase: Phase,
    lost_at: Option<Instant>,
    /// Whether borrow.cfg is written
    configured: bool,
}

impl Default for Borrow {
    fn default() -> Self {
        Self {
            phase: Phase::Idle,
            lost_at: None,
            configured: false,
        }
    }
}

impl Borrow {
    pub fn borrow(&mut self) {
        if self.phase == Phase::Idle {
            self.phase = Phase::Connecting;
        }
    }

    /// The control connection is up and holds the lease
    pub fn connected(&mut self) -> Option<Action> {
        self.phase = Phase::Borrowed;
        self.lost_at = None;
        self.configured = true;
        // Written again: a lease taken anew has a new token
        Some(Action::WriteConfig)
    }

    /// A heartbeat or a connection attempt failed
    pub fn failed(&mut self, now: Instant) -> Option<Action> {
        match self.phase {
            Phase::Borrowed => {
                self.phase = Phase::Reconnecting;
                self.lost_at = Some(now);
                None
            }
            Phase::Reconnecting => self.tick(now),
            Phase::Idle => None,
            Phase::Connecting | Phase::NoHost | Phase::Refused(_) => {
                self.phase = Phase::NoHost;
                None
            }
        }
    }

    /// The lender answered but refused the lease
    pub fn refused(&mut self, reason: String, now: Instant) -> Option<Action> {
        let action = self.failed(now);
        if self.phase != Phase::Reconnecting {
            self.phase = Phase::Refused(reason);
        }
        action
    }

    /// Time passes: the grace may run out
    pub fn tick(&mut self, now: Instant) -> Option<Action> {
        if self.phase == Phase::Reconnecting
            && self.lost_at.is_some_and(|lost| now.saturating_duration_since(lost) >= GRACE)
        {
            self.phase = Phase::NoHost;
            self.lost_at = None;
            return self.unconfigure();
        }
        None
    }

    /// The user returned the GPU
    pub fn give_back(&mut self) -> Option<Action> {
        self.phase = Phase::Idle;
        self.lost_at = None;
        self.unconfigure()
    }

    fn unconfigure(&mut self) -> Option<Action> {
        if self.configured {
            self.configured = false;
            Some(Action::RemoveConfig)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECOND: Duration = Duration::from_secs(1);

    #[test]
    fn borrowing_writes_the_config_and_returning_removes_it() {
        let mut borrow = Borrow::default();
        borrow.borrow();
        assert_eq!(borrow.phase, Phase::Connecting);
        assert_eq!(borrow.connected(), Some(Action::WriteConfig));
        assert_eq!(borrow.give_back(), Some(Action::RemoveConfig));
        assert_eq!(borrow.phase, Phase::Idle);
        assert_eq!(borrow.give_back(), None);
    }

    #[test]
    fn a_short_drop_keeps_the_config() {
        let start = Instant::now();
        let mut borrow = Borrow::default();
        borrow.borrow();
        borrow.connected();
        assert_eq!(borrow.failed(start), None);
        assert_eq!(borrow.phase, Phase::Reconnecting);
        // Retries keep failing within the grace
        assert_eq!(borrow.failed(start + 2 * SECOND), None);
        assert_eq!(borrow.tick(start + GRACE - SECOND), None);
        // Back: the config is written again, with the lease's token
        assert_eq!(borrow.connected(), Some(Action::WriteConfig));
        assert_eq!(borrow.phase, Phase::Borrowed);
        assert_eq!(borrow.tick(start + GRACE * 2), None);
    }

    #[test]
    fn a_long_drop_removes_the_config_then_borrows_again() {
        let start = Instant::now();
        let mut borrow = Borrow::default();
        borrow.borrow();
        borrow.connected();
        borrow.failed(start);
        assert_eq!(borrow.tick(start + GRACE), Some(Action::RemoveConfig));
        assert_eq!(borrow.phase, Phase::NoHost);
        assert_eq!(borrow.failed(start + GRACE + SECOND), None);
        assert_eq!(borrow.phase, Phase::NoHost);
        assert_eq!(borrow.connected(), Some(Action::WriteConfig));
    }

    #[test]
    fn a_refusal_shows_why() {
        let start = Instant::now();
        let mut borrow = Borrow::default();
        borrow.borrow();
        assert_eq!(borrow.refused("lent to OTHER".into(), start), None);
        assert_eq!(borrow.phase, Phase::Refused("lent to OTHER".into()));

        // Refused while reconnecting (someone took the lease after it ran out): the grace
        // still decides when the GPU is gone
        let mut borrow = Borrow::default();
        borrow.borrow();
        borrow.connected();
        borrow.failed(start);
        assert_eq!(borrow.refused("lent to OTHER".into(), start + SECOND), None);
        assert_eq!(borrow.phase, Phase::Reconnecting);
        assert_eq!(
            borrow.refused("lent to OTHER".into(), start + GRACE),
            Some(Action::RemoveConfig)
        );
        assert_eq!(borrow.phase, Phase::Refused("lent to OTHER".into()));
    }
}
