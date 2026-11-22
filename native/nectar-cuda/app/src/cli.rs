//! The nectar command, where there is no window yet (Linux, macOS):
//!
//!   nectar [status]          what is borrowed, and the lenders found
//!   nectar borrow <lender>   borrow a lender's GPU; programs then find it on their own
//!   nectar return            give it back
//!   nectar add <address>     look for a lender at an address, beyond the local network
//!   nectar profile [gaming | ai]   what the GPU is for, which tunes the link for it
//!   nectar link              measures the link to the lender, with advice for the profile
//!   nectar detect            this computer's GPUs and the APIs their drivers offer
//!   nectar lend              lend this computer's GPU: runs nectar-server --lend with the
//!                            secret from NECTAR_LEND_SECRET, or asks for it
//!
//! NECTAR_LEND_LISTEN moves the lender off 0.0.0.0, such as onto loopback for tests.

use std::io::{self, BufRead, Write};
use std::process::{self, Command};
use std::{env, thread, time::Duration};

use crate::service;

pub fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let result = match args[..] {
        [] | ["status"] => status(),
        ["borrow", lender] => send(&format!("borrow {lender}")).and_then(|()| wait_for_borrow()),
        ["return"] => send("return").map(|()| println!("Returned")),
        ["add", address] => send(&format!("add {address}"))
            .map(|()| println!("Looking for a lender at {address}; it shows in `nectar status` once it answers")),
        ["profile"] => service::status().map(|status| println!("{}", status.profile)),
        ["profile", profile] => send(&format!("profile {profile}"))
            .map(|()| println!("Programs started from now on are tuned for {profile}")),
        ["link"] => link(),
        ["detect"] => {
            print!("{}", nectar_detect::inventory().report());
            Ok(())
        }
        ["lend"] => lend(),
        _ => {
            eprintln!("usage: nectar [status] | borrow <lender> | return | add <address> | profile [gaming | ai] | link | detect | lend");
            process::exit(2);
        }
    };
    if let Err(error) = result {
        eprintln!("nectar: {error}");
        if error.kind() == io::ErrorKind::NotFound || error.kind() == io::ErrorKind::ConnectionRefused {
            eprintln!(
                "Is nectar-svc running? It listens on {}",
                nectar_lending::service_address().display()
            );
        }
        process::exit(1);
    }
}

fn send(command: &str) -> io::Result<()> {
    service::ask(command).map(|_| ())
}

fn status() -> io::Result<()> {
    let status = service::status()?;
    println!("{}", if status.message.is_empty() { &status.state } else { &status.message });
    println!("  tuned for {}{}", status.profile, if status.wifi_held { ", Wi-Fi in low-latency mode" } else { "" });
    if !status.note.is_empty() {
        println!("  {}", status.note);
    }
    if status.state == "borrowed" {
        println!(
            "  {:.1} ms round trip, {:.0} calls/s, {:.1} MB/s up, {:.1} MB/s down",
            status.rtt_ms, status.calls_per_s, status.up_mb_s, status.down_mb_s
        );
        if let Some(percent) = status.gpu_percent {
            println!("  the lender's GPU is {percent}% busy");
        }
    }
    if status.found.is_empty() {
        println!("\nNo lenders found yet. On another network, try `nectar add <address>`.");
        return Ok(());
    }
    println!("\n{:<18} {:<32} {:>8}  {:<22} {:<20} {:>8}", "Lender", "GPU", "VRAM", "Lends", "State", "Round trip");
    for lender in &status.found {
        let state = if lender.state == "lent" {
            format!("lent to {}", lender.borrower)
        } else {
            "ready".to_string()
        };
        println!(
            "{:<18} {:<32} {:>5.1} GB  {:<22} {:<20} {:>5} ms",
            lender.name,
            lender.gpu,
            lender.vram_mib.parse::<f64>().unwrap_or(0.0) / 1024.0,
            lender.apis,
            state,
            lender.rtt_ms
        );
    }
    Ok(())
}

/// Runs the link check and shows what it found
fn link() -> io::Result<()> {
    send("check")?;
    println!("Measuring the link for a few seconds...");
    loop {
        thread::sleep(Duration::from_millis(500));
        let status = service::status()?;
        if status.check != "running" {
            for line in status.check_lines.iter().chain(&status.advice) {
                println!("{line}");
            }
            return Ok(());
        }
    }
}

/// Shows how the borrow goes, for up to 10 s
fn wait_for_borrow() -> io::Result<()> {
    let mut last = String::new();
    for _ in 0..50 {
        let status = service::status()?;
        if status.message != last {
            println!("{}", status.message);
            last = status.message.clone();
        }
        if matches!(status.state.as_str(), "borrowed" | "refused") {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(200));
    }
    Ok(())
}

fn lend() -> io::Result<()> {
    let secret = match env::var("NECTAR_LEND_SECRET") {
        Ok(secret) if !secret.is_empty() => secret,
        _ => {
            print!("Shared secret (shown as you type): ");
            io::stdout().flush()?;
            let mut line = String::new();
            io::stdin().lock().read_line(&mut line)?;
            line.trim().to_string()
        }
    };
    if secret.len() < 8 {
        return Err(io::Error::other("the secret needs at least 8 characters"));
    }
    let server = env::current_exe()?
        .parent()
        .map(|folder| folder.join("nectar-server"))
        .unwrap_or_default();
    let listen = env::var("NECTAR_LEND_LISTEN")
        .unwrap_or_else(|_| format!("0.0.0.0:{}", nectar_cuda_protocol::DEFAULT_PORT));
    let status = Command::new(&server)
        .args(["--lend", "--listen", &listen])
        // Not on the command line, where other users could read it
        .env("NECTAR_LEND_SECRET", secret)
        .status()
        .map_err(|error| io::Error::new(error.kind(), format!("{}: {error}", server.display())))?;
    if !status.success() {
        return Err(io::Error::other(format!("nectar-server stopped: {status}")));
    }
    Ok(())
}
