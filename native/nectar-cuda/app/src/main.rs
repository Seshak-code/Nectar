//! nectar: lend this computer's GPU, or borrow one from another computer on the network.
//!
//! On Windows it is a window and a tray icon (gui.rs). Elsewhere it is, for now, a command
//! (cli.rs). Both talk to nectar-svc (service.rs).

#![cfg_attr(windows, windows_subsystem = "windows")]

#[cfg(not(windows))]
mod cli;
#[cfg(windows)]
mod gui;
#[cfg(windows)]
mod lender;
mod service;

fn main() {
    #[cfg(windows)]
    gui::main();
    #[cfg(not(windows))]
    cli::main();
}
