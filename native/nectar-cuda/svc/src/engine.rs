//! What nectar-svc does: finds lenders, holds the lease with a control connection, keeps
//! the hybrid runtime's sandbox filled, and writes borrow.cfg while a GPU is borrowed.

use std::collections::HashMap;
use std::fs;
use std::io::{self, BufReader, BufWriter, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs, UdpSocket};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use nectar_cuda_protocol::{self as protocol, Reader};
use nectar_lending::discovery::{self, LenderInfo, State};
use nectar_lending::{BorrowConfig, Profile, crypto, link, wifi};

use crate::borrow::{Action, Borrow, Phase};
use crate::log;

/// How often lenders are probed, and when one not heard from is dropped from the list
const PROBE_EVERY: Duration = Duration::from_secs(2);
const FORGET_AFTER: Duration = Duration::from_secs(10);
/// Heartbeats on the control connection, and how long one may take
const BEAT_EVERY: Duration = Duration::from_secs(1);
const BEAT_TIMEOUT: Duration = Duration::from_secs(3);
/// Between attempts to reach a lender that is away
const RETRY_EVERY: Duration = Duration::from_secs(2);
/// The APIs this computer has a Nectar driver for, and so borrows when a lender offers them
const BORROWABLE: &[&str] = &["cuda", "vulkan"];

/// A lender discovery found
#[derive(Clone)]
pub struct Found {
    pub address: SocketAddr,
    pub info: LenderInfo,
    pub seen: Instant,
    pub rtt: Duration,
}

/// What the app shows
#[derive(Clone, Default)]
pub struct Status {
    pub phase: String,
    pub message: String,
    pub lender: Option<Found>,
    pub rtt_ms: f64,
    pub calls_per_s: f64,
    pub up_mb_s: f64,
    pub down_mb_s: f64,
    /// How busy the lender's GPU is, in percent, when it can say
    pub gpu_percent: Option<u32>,
}

/// The link check, as the app shows it
#[derive(Clone, Default)]
struct Check {
    /// running, done or failed; empty when none has run
    state: String,
    lines: Vec<String>,
    advice: Vec<String>,
}

struct Shared {
    /// By lender name
    found: HashMap<String, Found>,
    /// Addresses to probe directly, for networks that drop broadcasts
    manual: Vec<String>,
    /// The lender the user chose to borrow from, by name
    wanted: Option<String>,
    /// What the user chose the GPU for
    profile: Profile,
    /// Whether this computer's Wi-Fi is being held in low-latency mode
    wifi_held: bool,
    check: Check,
    stop: bool,
    status: Status,
}

pub struct Engine {
    /// This engine, for the threads it starts
    me: Mutex<std::sync::Weak<Engine>>,
    secret: Vec<u8>,
    shared: Mutex<Shared>,
    wake: Condvar,
}

impl Engine {
    pub fn new(secret: Vec<u8>) -> Arc<Self> {
        let engine = Arc::new(Self {
            me: Mutex::new(std::sync::Weak::new()),
            secret,
            shared: Mutex::new(Shared {
                found: HashMap::new(),
                manual: Vec::new(),
                wanted: None,
                profile: Profile::saved(),
                wifi_held: false,
                check: Check::default(),
                stop: false,
                status: Status {
                    phase: "idle".into(),
                    ..Status::default()
                },
            }),
            wake: Condvar::new(),
        });
        *engine.me.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Arc::downgrade(&engine);
        engine
    }

    fn shared(&self) -> MutexGuard<'_, Shared> {
        self.shared
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Starts finding lenders and borrowing; the handle ends after `stop`
    pub fn start(self: &Arc<Self>) -> JoinHandle<()> {
        // A borrow.cfg left by a crash names a lease this service no longer holds
        if let Err(error) = BorrowConfig::remove() {
            log(&format!("cannot remove a leftover borrow.cfg: {error}"));
        }
        let engine = Arc::clone(self);
        thread::spawn(move || engine.discover());
        let engine = Arc::clone(self);
        thread::spawn(move || engine.lend_loop())
    }

    /// Returns the GPU if one is borrowed, and ends
    pub fn stop(&self) {
        self.shared().stop = true;
        self.wake.notify_all();
    }

    /// Carries out a command from the app (pipe.rs) and returns its answer
    pub fn command(&self, line: &str) -> String {
        let (verb, argument) = line.trim().split_once(' ').unwrap_or((line.trim(), ""));
        match verb {
            "status" => self.status_text(),
            "borrow" => {
                let mut shared = self.shared();
                if !shared.found.contains_key(argument) {
                    return format!("error=no lender named {argument} has answered\n");
                }
                shared.wanted = Some(argument.to_string());
                drop(shared);
                self.wake.notify_all();
                "ok=1\n".into()
            }
            "return" => {
                self.shared().wanted = None;
                self.wake.notify_all();
                "ok=1\n".into()
            }
            "profile" => match Profile::parse(argument) {
                Some(profile) => {
                    // A check's advice was for the other profile
                    let mut shared = self.shared();
                    shared.profile = profile;
                    if shared.check.state != "running" {
                        shared.check = Check::default();
                    }
                    drop(shared);
                    if let Err(error) = profile.save() {
                        log(&format!("cannot keep the profile: {error}"));
                    }
                    self.wake.notify_all();
                    "ok=1\n".into()
                }
                None => format!("error=no profile named {argument}\n"),
            },
            "check" => self.start_check(),
            "add" if !argument.is_empty() => {
                let mut shared = self.shared();
                if !shared.manual.iter().any(|host| host == argument) {
                    shared.manual.push(argument.to_string());
                }
                "ok=1\n".into()
            }
            _ => format!("error=unknown command {verb}\n"),
        }
    }

    /// Starts a link check on a thread of its own; the status follows how it goes
    fn start_check(&self) -> String {
        let Some(config) = BorrowConfig::read() else {
            return "error=borrow a GPU first: the check measures the link to its lender\n".into();
        };
        let mut shared = self.shared();
        if shared.check.state == "running" {
            return "ok=1\n".into();
        }
        let profile = shared.profile;
        shared.check = Check { state: "running".into(), ..Check::default() };
        drop(shared);
        let engine = self.me.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).upgrade();
        thread::spawn(move || {
            let check = match link::check(&config.server, &config.token, profile) {
                Ok(report) => {
                    let check = Check { state: "done".into(), lines: report.lines(), advice: report.advice() };
                    log(&format!("link check ({}): {}", profile.key(), check.lines.join("; ")));
                    check
                }
                Err(error) => {
                    log(&format!("link check failed: {error}"));
                    Check { state: "failed".into(), lines: vec![format!("The check failed: {error}")], advice: Vec::new() }
                }
            };
            if let Some(engine) = engine {
                engine.shared().check = check;
            }
        });
        "ok=1\n".into()
    }

    /// key=value lines; one `found=` line per lender:
    /// name|address|gpu|vram MiB|CUDA|state|borrower|rtt ms
    fn status_text(&self) -> String {
        let shared = self.shared();
        let status = &shared.status;
        let mut text = format!(
            "state={}\nmessage={}\nrtt_ms={:.1}\ncalls_per_s={:.0}\nup_mb_s={:.1}\ndown_mb_s={:.1}\n",
            status.phase, status.message, status.rtt_ms, status.calls_per_s, status.up_mb_s, status.down_mb_s
        );
        text += &format!("profile={}\nwifi_held={}\n", shared.profile.key(), shared.wifi_held as u8);
        if let Some(percent) = status.gpu_percent {
            text += &format!("gpu_percent={percent}\n");
        }
        if let Some(lender) = &status.lender {
            text += &format!("lender={}\ngpu={}\n", lender.info.name, lender.info.gpu);
            let apis: Vec<String> = lender.info.apis.iter().map(|offer| offer.api.clone()).collect();
            if let Some(note) = shared.profile.note(&apis) {
                text += &format!("note={note}\n");
            }
        }
        if !shared.check.state.is_empty() {
            text += &format!("check={}\n", shared.check.state);
            for line in &shared.check.lines {
                text += &format!("check_line={line}\n");
            }
            for line in &shared.check.advice {
                text += &format!("advice={line}\n");
            }
        }
        let mut found: Vec<&Found> = shared.found.values().collect();
        found.sort_by(|a, b| a.info.name.cmp(&b.info.name));
        for lender in found {
            let state = match lender.info.state {
                State::Available => "available",
                State::Lent => "lent",
            };
            text += &format!(
                "found={}|{}|{}|{}|{}|{}|{}|{:.1}\n",
                lender.info.name,
                lender.address,
                lender.info.gpu,
                lender.info.vram_mib,
                lender.info.lends(),
                state,
                lender.info.borrower,
                lender.rtt.as_secs_f64() * 1000.0
            );
        }
        text
    }

    /// Probes for lenders every PROBE_EVERY, by broadcast, on this machine, and at the
    /// addresses added by hand
    fn discover(self: Arc<Self>) {
        let socket = match UdpSocket::bind("0.0.0.0:0").and_then(|socket| {
            socket.set_broadcast(true)?;
            socket.set_read_timeout(Some(Duration::from_millis(200)))?;
            Ok(socket)
        }) {
            Ok(socket) => socket,
            Err(error) => {
                log(&format!("cannot look for lenders: {error}"));
                return;
            }
        };

        let mut buffer = [0u8; 2048];
        while !self.shared().stop {
            let now = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
            let (probe, nonce) = discovery::probe(&self.secret, now);
            let mut targets = vec![
                SocketAddr::from(([255, 255, 255, 255], discovery::PORT)),
                SocketAddr::from(([127, 0, 0, 1], discovery::PORT)),
            ];
            for host in self.shared().manual.clone() {
                let host = host.split(':').next().unwrap_or_default().to_string();
                if let Ok(addresses) = (host.as_str(), discovery::PORT).to_socket_addrs() {
                    targets.extend(addresses.filter(SocketAddr::is_ipv4));
                }
            }
            let sent = Instant::now();
            for target in &targets {
                let _ = socket.send_to(probe.as_bytes(), target);
            }

            while sent.elapsed() < PROBE_EVERY {
                let Ok((length, from)) = socket.recv_from(&mut buffer) else {
                    continue;
                };
                let Some(info) = std::str::from_utf8(&buffer[..length])
                    .ok()
                    .and_then(|text| discovery::check_answer(&self.secret, text, &nonce))
                else {
                    continue;
                };
                let found = Found {
                    address: SocketAddr::new(from.ip(), info.port),
                    rtt: sent.elapsed(),
                    seen: Instant::now(),
                    info,
                };
                let mut shared = self.shared();
                // The same lender answers on loopback and on the network: keep the network
                if let Some(known) = shared.found.get(&found.info.name)
                    && !known.address.ip().is_loopback()
                    && found.address.ip().is_loopback()
                    && known.seen.elapsed() < FORGET_AFTER
                {
                    continue;
                }
                shared.found.insert(found.info.name.clone(), found);
            }
            self.shared().found.retain(|_, found| found.seen.elapsed() < FORGET_AFTER);
        }
    }

    /// Borrows the lender the user chose, keeps the lease alive, and follows its state
    fn lend_loop(self: Arc<Self>) {
        let mut borrow = Borrow::default();
        let mut control: Option<Control> = None;
        let mut config: Option<BorrowConfig> = None;
        let mut last_try: Option<Instant> = None;
        let mut last_totals: Option<(Instant, [u64; 3])> = None;

        loop {
            let (wanted, stop) = {
                let shared = self.shared();
                let shared = self
                    .wake
                    .wait_timeout(shared, BEAT_EVERY)
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .0;
                (shared.wanted.clone(), shared.stop)
            };
            let now = Instant::now();

            // Returned, stopped, or a different lender chosen
            let keep = wanted.is_some()
                && !stop
                && config.as_ref().is_none_or(|config| Some(&config.lender) == wanted.as_ref());
            if !keep {
                if let Some(mut lease) = control.take() {
                    lease.give_back();
                }
                self.apply(borrow.give_back(), &config);
                config = None;
                last_totals = None;
                self.shared().wifi_held = false;
            }
            if stop {
                return;
            }
            let Some(name) = wanted else {
                self.publish(&borrow, &config, None);
                continue;
            };

            // A new choice: programs started from now on follow it, and so does the lender
            let profile = self.shared().profile;
            if let Some(current) = config.as_mut().filter(|current| current.profile != profile) {
                current.profile = profile;
                if borrow.phase == Phase::Borrowed {
                    self.apply(Some(Action::WriteConfig), &config);
                }
                if let Some(lease) = &mut control {
                    lease.choose(profile);
                }
            }

            match &mut control {
                Some(lease) => match lease.beat() {
                    Ok((rtt, totals, gpu_percent)) => {
                        let mut status = self.shared().status.clone();
                        status.rtt_ms = rtt.as_secs_f64() * 1000.0;
                        status.gpu_percent = gpu_percent;
                        if let Some((at, before)) = last_totals {
                            let seconds = now.duration_since(at).as_secs_f64().max(0.001);
                            status.calls_per_s = totals[0].saturating_sub(before[0]) as f64 / seconds;
                            status.up_mb_s = totals[1].saturating_sub(before[1]) as f64 / 1e6 / seconds;
                            status.down_mb_s = totals[2].saturating_sub(before[2]) as f64 / 1e6 / seconds;
                        }
                        last_totals = Some((now, totals));
                        self.shared().status = status;
                    }
                    Err(error) => {
                        log(&format!("lost the lender {name}: {error}"));
                        control = None;
                        last_totals = None;
                        self.apply(borrow.failed(now), &config);
                    }
                },
                None if last_try.is_none_or(|at| now.duration_since(at) >= RETRY_EVERY) => {
                    last_try = Some(now);
                    borrow.borrow();
                    let found = self.shared().found.get(&name).cloned();
                    match found {
                        None => {
                            self.apply(borrow.failed(now), &config);
                        }
                        Some(found) => match Control::connect(&found, &self.secret) {
                            Ok(mut lease) => {
                                // The hybrid runtime is a Windows DLL for one CPU
                                let driver = if cfg!(windows) && found.info.same_platform() {
                                    ensure_hybrid(&mut lease, &found.info.hybrid)
                                } else {
                                    String::new()
                                };
                                config = Some(BorrowConfig {
                                    server: found.address.to_string(),
                                    token: lease.token.clone(),
                                    driver,
                                    lender: found.info.name.clone(),
                                    apis: found
                                        .info
                                        .apis
                                        .iter()
                                        .map(|offer| offer.api.clone())
                                        .filter(|api| BORROWABLE.contains(&api.as_str()))
                                        .collect(),
                                    profile,
                                });
                                lease.choose(profile);
                                self.shared().wifi_held = lease.wifi.is_some();
                                log(&format!("borrowed {} from {}", found.info.gpu, found.info.name));
                                control = Some(lease);
                                self.apply(borrow.connected(), &config);
                            }
                            Err(error) if error.kind() == io::ErrorKind::PermissionDenied => {
                                self.apply(borrow.refused(error.to_string(), now), &config);
                            }
                            Err(error) => {
                                log(&format!("cannot reach {name}: {error}"));
                                self.apply(borrow.failed(now), &config);
                            }
                        },
                    }
                }
                None => {}
            }

            self.apply(borrow.tick(now), &config);
            let lender = self.shared().found.get(&name).cloned();
            self.publish(&borrow, &config, lender);
        }
    }

    fn apply(&self, action: Option<Action>, config: &Option<BorrowConfig>) {
        let outcome = match (action, config) {
            (Some(Action::WriteConfig), Some(config)) => config.write(),
            (Some(Action::RemoveConfig), _) => BorrowConfig::remove(),
            _ => Ok(()),
        };
        if let Err(error) = outcome {
            log(&format!("cannot update {}: {error}", BorrowConfig::path().display()));
        }
    }

    fn publish(&self, borrow: &Borrow, config: &Option<BorrowConfig>, lender: Option<Found>) {
        let name = config.as_ref().map(|config| config.lender.clone()).unwrap_or_default();
        let (phase, message) = match &borrow.phase {
            Phase::Idle => ("idle", "No GPU borrowed".to_string()),
            Phase::Connecting => ("connecting", "Connecting...".to_string()),
            Phase::Borrowed => ("borrowed", format!("Using the GPU on {name}")),
            Phase::Reconnecting => ("reconnecting", format!("Reconnecting to {name}...")),
            Phase::NoHost => ("nohost", "No host available".to_string()),
            Phase::Refused(reason) => ("refused", format!("Refused: {reason}")),
        };
        let mut shared = self.shared();
        shared.status.phase = phase.into();
        shared.status.message = message;
        shared.status.lender = lender;
        if borrow.phase != Phase::Borrowed {
            shared.status.calls_per_s = 0.0;
            shared.status.up_mb_s = 0.0;
            shared.status.down_mb_s = 0.0;
            shared.status.gpu_percent = None;
        }
    }
}

/// The lease's control connection
struct Control {
    reader: BufReader<TcpStream>,
    writer: BufWriter<TcpStream>,
    token: String,
    /// This computer's Wi-Fi in low-latency mode, for as long as the lease's connection lasts
    wifi: Option<wifi::Hold>,
}

impl Control {
    /// Connects, proves the secret and takes the lease. A refusal is an error of kind
    /// PermissionDenied with the lender's reason.
    fn connect(found: &Found, secret: &[u8]) -> io::Result<Self> {
        let stream = TcpStream::connect_timeout(&found.address, Duration::from_secs(2))?;
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(BEAT_TIMEOUT))?;
        let held = wifi::hold(stream.local_addr()?.ip(), found.address.ip());
        if held.is_some() {
            log("Wi-Fi is in low-latency mode while the GPU is borrowed");
        }
        let mut reader = BufReader::new(stream.try_clone()?);
        let mut writer = BufWriter::new(stream);
        protocol::write_hello(&mut writer, protocol::HELLO_CONTROL)?;
        let (_, challenge) = protocol::read_lender_hello(&mut reader)?;
        if challenge.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "this server does not lend (it was started without --lend)",
            ));
        }
        protocol::write_proof(
            &mut writer,
            &crypto::hmac(secret, &challenge),
            &nectar_lending::computer_name(),
        )?;
        let token = protocol::read_verdict(&mut reader)?;
        Ok(Self {
            reader,
            writer,
            token,
            wifi: held,
        })
    }

    fn request(&mut self, function: u32) -> io::Result<Vec<u8>> {
        protocol::write_message(&mut self.writer, function, &[])?;
        self.writer.flush()?;
        match protocol::read_message(&mut self.reader)? {
            (0, results) => Ok(results),
            (result, _) => Err(io::Error::other(format!("the lender answered {result}"))),
        }
    }

    /// A heartbeat: its round trip, the lender's totals of calls, bytes received and bytes sent
    /// for the lease, and how busy its GPU is
    fn beat(&mut self) -> io::Result<(Duration, [u64; 3], Option<u32>)> {
        let sent = Instant::now();
        let results = self.request(protocol::BEAT)?;
        let rtt = sent.elapsed();
        let mut reader = Reader::new(&results);
        let totals = [reader.u64()?, reader.u64()?, reader.u64()?];
        Ok((rtt, totals, reader.u32().ok().filter(|&percent| percent != u32::MAX)))
    }

    /// Tells the lender what the GPU is for
    fn choose(&mut self, profile: Profile) {
        let sent = protocol::write_message(&mut self.writer, protocol::PROFILE, &[profile.key().as_bytes()])
            .and_then(|()| self.writer.flush())
            .and_then(|()| protocol::read_message(&mut self.reader));
        if let Err(error) = sent {
            log(&format!("cannot tell the lender the profile: {error}"));
        }
    }

    fn give_back(&mut self) {
        let _ = self.request(protocol::RETURN);
    }
}

/// Makes sure the sandbox has the lender's hybrid runtime, fetching it if not. Returns its
/// key, or an empty one if the lender has none to give.
fn ensure_hybrid(lease: &mut Control, key: &str) -> String {
    if key.is_empty() {
        return String::new();
    }
    let path = nectar_lending::driver_folder(key).join(nectar_lending::HYBRID);
    if path.exists() {
        return key.to_string();
    }
    let fetched = lease.request(protocol::HYBRID).and_then(|results| {
        let mut reader = Reader::new(&results);
        let key = String::from_utf8_lossy(reader.bytes()?).into_owned();
        let bytes = reader.bytes()?;
        let folder = nectar_lending::driver_folder(&key);
        fs::create_dir_all(&folder)?;
        let part = folder.join("download.part");
        fs::write(&part, bytes)?;
        fs::rename(&part, folder.join(nectar_lending::HYBRID))?;
        Ok(key)
    });
    match fetched {
        Ok(key) => {
            log(&format!("received the lender's hybrid runtime {key}"));
            key
        }
        Err(error) => {
            log(&format!("cannot fetch the lender's hybrid runtime: {error}"));
            String::new()
        }
    }
}
