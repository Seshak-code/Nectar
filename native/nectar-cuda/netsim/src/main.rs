//! nectar-netsim: a simulated Wi-Fi link on one computer. It sits between a borrower and a lender
//! as a TCP proxy, and holds each direction's bytes back as a link would: limited rate, delay,
//! jitter, the stalls that lost packets cause, and a signal that weakens and fades. It works the
//! same on Windows and on Linux, and needs no administrator (WSL's `tc` would).
//!
//!   nectar-netsim --listen 127.0.0.1:43300 --to 127.0.0.1:43212 --profile measured
//!   nectar-netsim --to ... --profile measured --attenuation-db 15 --fade 8,10
//!   nectar-netsim check 127.0.0.1:43300     measures it as a borrower would
//!   nectar-netsim profiles                  lists the profiles
//!
//! Point a program at --listen with NECTAR_VK_SERVER or NECTAR_CUDA_SERVER. Nectar treats loopback
//! as no network, so give it NECTAR_VK_KEEP_UPLOADS=1, NECTAR_VK_PACK=1 and NECTAR_CUDA_KEEP_UPLOADS=1 to act as it
//! does over Wi-Fi.
//!
//! This is a model, not a radio. `measured` is what the real laptop and desktop measured; the other
//! profiles and the effect of attenuation are estimates (model.rs).

mod model;

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use std::{env, process, thread};

use model::{Air, Link, Rng};

/// Bytes handled as one packet on the air
const PACKET: usize = 16 << 10;
/// The most waiting to go in one direction, as a card's and a router's queues hold
const QUEUE_LIMIT: usize = 4 << 20;

/// One direction's queue: packets with the time they arrive
#[derive(Default)]
struct Queue {
    packets: VecDeque<(Instant, Vec<u8>)>,
    bytes: usize,
    closed: bool,
}

struct Channel {
    queue: Mutex<Queue>,
    changed: Condvar,
}

/// Waits until `when`: sleeps most of the way, and yields for the last of it, as a sleep can
/// overrun by a millisecond even with the system's timer at its finest
fn wait_until(when: Instant) {
    loop {
        let now = Instant::now();
        if now >= when {
            return;
        }
        let left = when - now;
        if left > Duration::from_millis(2) {
            thread::sleep(left - Duration::from_millis(2));
        } else {
            thread::yield_now();
        }
    }
}

/// Reads from `from`, and schedules each packet on the air
fn read_side(mut from: TcpStream, channel: Arc<Channel>, air: Arc<Air>, up: bool) {
    let mut buffer = vec![0u8; PACKET];
    let mut rng = Rng::new(if up { 0x1234_5678 } else { 0x8765_4321 } ^ std::process::id() as u64);
    loop {
        let count = match from.read(&mut buffer) {
            Ok(0) | Err(_) => break,
            Ok(count) => count,
        };
        let due = air.schedule(up, count, Instant::now(), &mut rng);
        let mut queue = channel.queue.lock().unwrap_or_else(|p| p.into_inner());
        while queue.bytes > QUEUE_LIMIT && !queue.closed {
            queue = channel.changed.wait(queue).unwrap_or_else(|p| p.into_inner());
        }
        if queue.closed {
            break;
        }
        queue.bytes += count;
        queue.packets.push_back((due, buffer[..count].to_vec()));
        channel.changed.notify_all();
    }
    channel.queue.lock().unwrap_or_else(|p| p.into_inner()).closed = true;
    channel.changed.notify_all();
}

/// Writes each packet out when it arrives
fn write_side(mut to: TcpStream, channel: Arc<Channel>) {
    loop {
        let (due, packet) = {
            let mut queue = channel.queue.lock().unwrap_or_else(|p| p.into_inner());
            loop {
                if let Some(next) = queue.packets.pop_front() {
                    queue.bytes -= next.1.len();
                    channel.changed.notify_all();
                    break next;
                }
                if queue.closed {
                    let _ = to.shutdown(std::net::Shutdown::Write);
                    return;
                }
                queue = channel.changed.wait(queue).unwrap_or_else(|p| p.into_inner());
            }
        };
        wait_until(due);
        if to.write_all(&packet).is_err() {
            let mut queue = channel.queue.lock().unwrap_or_else(|p| p.into_inner());
            queue.closed = true;
            channel.changed.notify_all();
            return;
        }
    }
}

fn direction(from: TcpStream, to: TcpStream, air: &Arc<Air>, up: bool) {
    let channel = Arc::new(Channel { queue: Mutex::new(Queue::default()), changed: Condvar::new() });
    let (reading, writing) = (Arc::clone(&channel), channel);
    let air = Arc::clone(air);
    thread::spawn(move || read_side(from, reading, air, up));
    thread::spawn(move || write_side(to, writing));
}

fn proxy(listen: &str, target: &str, air: Arc<Air>) -> std::io::Result<()> {
    let listener = TcpListener::bind(listen)?;
    eprintln!("nectar-netsim: {listen} to {target}, {}", air.describe());
    for borrower in listener.incoming() {
        let Ok(borrower) = borrower else { continue };
        let lender = match TcpStream::connect(target) {
            Ok(lender) => lender,
            Err(error) => {
                eprintln!("nectar-netsim: cannot reach {target}: {error}");
                continue;
            }
        };
        let _ = borrower.set_nodelay(true);
        let _ = lender.set_nodelay(true);
        direction(borrower.try_clone()?, lender.try_clone()?, &air, true);
        direction(lender, borrower, &air, false);
    }
    Ok(())
}

fn usage() -> ! {
    eprintln!(
        "usage: nectar-netsim --to <lender> [--listen <address>] [--profile <name>] [--attenuation-db <dB>] [--fade <seconds>,<dB>]\n       \
         nectar-netsim check <address>\n       nectar-netsim profiles"
    );
    process::exit(2);
}

/// Measures a link as a borrower does
fn check(address: &str) {
    match nectar_lending::link::check(address, "", nectar_lending::Profile::Ai) {
        Ok(report) => {
            for line in report.lines().iter().take(3) {
                println!("{line}");
            }
        }
        Err(error) => {
            eprintln!("nectar-netsim: {error}");
            process::exit(1);
        }
    }
}

fn main() {
    #[cfg(windows)]
    {
        // For this process only, so that short sleeps are short
        unsafe { windows_sys::Win32::Media::timeBeginPeriod(1) };
    }
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("profiles") => {
            for link in Link::ALL {
                println!("{:<14} {}", link.name, link.about());
            }
            return;
        }
        Some("check") => {
            let Some(address) = args.get(1) else { usage() };
            check(address);
            return;
        }
        _ => {}
    }
    let mut listen = "127.0.0.1:43300".to_string();
    let mut target = None;
    let mut link = Link::by_name("measured").expect("the measured profile");
    let (mut attenuation, mut fade) = (0.0, None);
    let mut rest = args.iter();
    while let Some(flag) = rest.next() {
        let Some(value) = rest.next() else { usage() };
        match flag.as_str() {
            "--listen" => listen = value.clone(),
            "--to" => target = Some(value.clone()),
            "--profile" => link = Link::by_name(value).unwrap_or_else(|| usage()),
            "--attenuation-db" => attenuation = value.parse().unwrap_or_else(|_| usage()),
            "--fade" => {
                let Some((period, depth)) = value.split_once(',') else { usage() };
                fade = Some((period.parse().unwrap_or_else(|_| usage()), depth.parse().unwrap_or_else(|_| usage())));
            }
            _ => usage(),
        }
    }
    let Some(target) = target else { usage() };
    if let Err(error) = proxy(&listen, &target, Arc::new(Air::new(link, attenuation, fade))) {
        eprintln!("nectar-netsim: {error}");
        process::exit(1);
    }
}
