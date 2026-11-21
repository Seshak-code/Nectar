//! Talking to nectar-svc over its pipe (Windows) or socket (elsewhere), as
//! nectar_lending::service_address names it: one line out, key=value lines and a blank line
//! back.

use std::io::{self, BufRead, BufReader, Read, Write};

/// A connection to the service, as a reader and a writer
fn connect() -> io::Result<(impl Read, impl Write)> {
    let address = nectar_lending::service_address();
    #[cfg(windows)]
    let stream = std::fs::OpenOptions::new().read(true).write(true).open(address)?;
    #[cfg(unix)]
    let stream = std::os::unix::net::UnixStream::connect(address)?;
    let writer = stream.try_clone()?;
    Ok((stream, writer))
}

/// A lender the service found
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lender {
    pub name: String,
    pub address: String,
    pub gpu: String,
    pub vram_mib: String,
    pub cuda: String,
    pub state: String,
    pub borrower: String,
    pub rtt_ms: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Status {
    /// idle, connecting, borrowed, reconnecting, nohost or refused
    pub state: String,
    pub message: String,
    pub rtt_ms: f64,
    pub calls_per_s: f64,
    pub up_mb_s: f64,
    pub down_mb_s: f64,
    pub found: Vec<Lender>,
}

/// Sends a command and returns the answer's key=value pairs
pub fn ask(command: &str) -> io::Result<Vec<(String, String)>> {
    let (reader, mut writer) = connect()?;
    writer.write_all(format!("{command}\n").as_bytes())?;
    let mut answer = Vec::new();
    for line in BufReader::new(reader).lines() {
        let line = line?;
        if line.is_empty() {
            break;
        }
        if let Some((key, value)) = line.split_once('=') {
            answer.push((key.to_string(), value.to_string()));
        }
    }
    if let Some((_, error)) = answer.iter().find(|(key, _)| key == "error") {
        return Err(io::Error::other(error.clone()));
    }
    Ok(answer)
}

pub fn status() -> io::Result<Status> {
    Ok(parse(&ask("status")?))
}

pub fn parse(answer: &[(String, String)]) -> Status {
    let mut status = Status::default();
    let number = |value: &str| value.parse().unwrap_or(0.0);
    for (key, value) in answer {
        match key.as_str() {
            "state" => status.state = value.clone(),
            "message" => status.message = value.clone(),
            "rtt_ms" => status.rtt_ms = number(value),
            "calls_per_s" => status.calls_per_s = number(value),
            "up_mb_s" => status.up_mb_s = number(value),
            "down_mb_s" => status.down_mb_s = number(value),
            "found" => {
                let parts: Vec<&str> = value.split('|').collect();
                if let [name, address, gpu, vram, cuda, state, borrower, rtt] = parts[..] {
                    status.found.push(Lender {
                        name: name.into(),
                        address: address.into(),
                        gpu: gpu.into(),
                        vram_mib: vram.into(),
                        cuda: cuda.into(),
                        state: state.into(),
                        borrower: borrower.into(),
                        rtt_ms: rtt.into(),
                    });
                }
            }
            _ => {}
        }
    }
    status
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_lines_parse() {
        let lines = "state=borrowed\nmessage=Using the GPU on DESKTOP\nrtt_ms=5.9\ncalls_per_s=120\n\
                     found=DESKTOP|192.168.1.89:43212|NVIDIA GeForce RTX 5070|12199|13.5|lent|LAPTOP|6.1\n";
        let answer: Vec<(String, String)> = lines
            .lines()
            .filter_map(|line| line.split_once('='))
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect();
        let status = parse(&answer);
        assert_eq!(status.state, "borrowed");
        assert_eq!(status.rtt_ms, 5.9);
        assert_eq!(status.found.len(), 1);
        assert_eq!(status.found[0].gpu, "NVIDIA GeForce RTX 5070");
        assert_eq!(status.found[0].borrower, "LAPTOP");
    }
}
