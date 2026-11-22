//! nectar-detect: lists this computer's GPUs, their kernel drivers, and the APIs their drivers
//! offer, official or community.

fn main() {
    print!("{}", nectar_detect::inventory().report());
}
