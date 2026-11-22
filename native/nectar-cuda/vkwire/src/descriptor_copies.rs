//! Descriptor copies, many together. Without descriptor buffers, which can't cross to the lender, vkd3d-proton
//! copies Direct3D 12's descriptors one at a time: The Witcher 3 makes about 27,000 of these calls a frame.
//! Each as a call of its own took the game's render thread about 640 ns, its share of the stream, and a
//! decode and a driver call on the lender, all for 7 numbers. So they wait and go together, in as few
//! bytes as they can, for one driver call.
//!
//! Elements mostly follow on from the last copy, so each goes as how far it is from there, in a number that's
//! short when that's small.
//!
//! ```text
//! DESCRIPTOR_COPIES: device, u32 count, then per copy
//!   a number: the copy's count times 2, plus 1 if the sets and bindings are new; if they are, u64 source
//!   set, u64 target set, u32 source binding, u32 target binding; then two signed numbers: how far the
//!   source and target elements are from where the last copy ended
//! ```

use std::io;

use nectar_cuda_protocol::{Reader, Writer};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DescriptorCopy {
    pub source_set: u64,
    pub target_set: u64,
    pub source_binding: u32,
    pub target_binding: u32,
    pub source_element: u32,
    pub target_element: u32,
    pub count: u32,
}

impl DescriptorCopy {
    fn same_places(&self, other: &Self) -> bool {
        (self.source_set, self.target_set, self.source_binding, self.target_binding)
            == (other.source_set, other.target_set, other.source_binding, other.target_binding)
    }
}

/// Adds a copy, joining it to the last if it carries straight on. Not within one set: a copy within a set
/// mustn't overlap itself, and joined ones might.
pub fn add(list: &mut Vec<DescriptorCopy>, copy: DescriptorCopy) {
    if let Some(last) = list.last_mut()
        && last.same_places(&copy)
        && last.source_set != last.target_set
        && last.source_element.checked_add(last.count) == Some(copy.source_element)
        && last.target_element.checked_add(last.count) == Some(copy.target_element)
        && let Some(count) = last.count.checked_add(copy.count)
    {
        last.count = count;
        return;
    }
    list.push(copy);
}

/// Base-128, low bits first
fn put_number(w: &mut Writer, mut value: u64) {
    while value >= 0x80 {
        w.u8(value as u8 | 0x80);
        value >>= 7;
    }
    w.u8(value as u8);
}

fn take_number(r: &mut Reader) -> io::Result<u64> {
    let (mut value, mut shift) = (0u64, 0);
    loop {
        let byte = r.u8()?;
        value |= ((byte & 0x7f) as u64) << shift;
        if byte < 0x80 {
            return Ok(value);
        }
        shift += 7;
        if shift > 35 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "a number too long"));
        }
    }
}

fn put_signed(w: &mut Writer, value: i64) {
    put_number(w, ((value << 1) ^ (value >> 63)) as u64);
}

fn take_signed(r: &mut Reader) -> io::Result<i64> {
    let value = take_number(r)?;
    Ok(((value >> 1) as i64) ^ -((value & 1) as i64))
}

pub fn encode(device: u64, list: &[DescriptorCopy], w: &mut Writer) {
    w.u64(device).u32(list.len() as u32);
    let mut before = DescriptorCopy::default();
    for (index, copy) in list.iter().enumerate() {
        let new = index == 0 || !before.same_places(copy);
        put_number(w, (copy.count as u64) << 1 | new as u64);
        if new {
            w.u64(copy.source_set).u64(copy.target_set).u32(copy.source_binding).u32(copy.target_binding);
        }
        put_signed(w, copy.source_element as i64 - (before.source_element as i64 + before.count as i64));
        put_signed(w, copy.target_element as i64 - (before.target_element as i64 + before.count as i64));
        before = *copy;
    }
}

/// The device, and how many copies follow
pub fn decode_head(r: &mut Reader) -> io::Result<(u64, u32)> {
    Ok((r.u64()?, r.u32()?))
}

/// The next copy, given the one before it
pub fn decode_next(r: &mut Reader, before: &DescriptorCopy) -> io::Result<DescriptorCopy> {
    let mut copy = *before;
    let head = take_number(r)?;
    copy.count = u32::try_from(head >> 1).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "a count too large"))?;
    if head & 1 != 0 {
        copy.source_set = r.u64()?;
        copy.target_set = r.u64()?;
        copy.source_binding = r.u32()?;
        copy.target_binding = r.u32()?;
    }
    let element = |from: u32, count: u32, far: i64| from as i64 + count as i64 + far;
    let source = element(before.source_element, before.count, take_signed(r)?);
    let target = element(before.target_element, before.count, take_signed(r)?);
    let bad = |_| io::Error::new(io::ErrorKind::InvalidData, "an element out of range");
    copy.source_element = u32::try_from(source).map_err(bad)?;
    copy.target_element = u32::try_from(target).map_err(bad)?;
    Ok(copy)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn copy(source_set: u64, target_set: u64, source_element: u32, target_element: u32) -> DescriptorCopy {
        DescriptorCopy { source_set, target_set, source_binding: 0, target_binding: 1, source_element, target_element, count: 1 }
    }

    #[test]
    fn copies_that_carry_straight_on_join() {
        let mut list = Vec::new();
        add(&mut list, copy(1, 2, 10, 20));
        add(&mut list, copy(1, 2, 11, 21));
        add(&mut list, copy(1, 2, 12, 22));
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].count, 3);
        // A gap in either, or other places
        add(&mut list, copy(1, 2, 14, 23));
        add(&mut list, copy(1, 2, 15, 25));
        add(&mut list, copy(1, 3, 16, 26));
        assert_eq!(list.len(), 4);
    }

    #[test]
    fn copies_within_a_set_stay_apart() {
        let mut list = Vec::new();
        add(&mut list, copy(1, 1, 10, 11));
        add(&mut list, copy(1, 1, 11, 12));
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn they_come_back_the_same() {
        let mut list = Vec::new();
        for at in 0..50 {
            add(&mut list, copy(1 + at as u64 / 10, 2, at * 3, at * 5));
        }
        let mut w = Writer::new();
        encode(99, &list, &mut w);
        // Not many bytes, as sets and bindings seldom change and elements are near where the last ended
        assert!(w.as_slice().len() < list.len() * 3 + 12 + 10 * 24, "{} bytes", w.as_slice().len());
        let mut r = Reader::new(w.as_slice());
        let (device, count) = decode_head(&mut r).unwrap();
        assert_eq!((device, count as usize), (99, list.len()));
        let mut before = DescriptorCopy::default();
        for expected in &list {
            before = decode_next(&mut r, &before).unwrap();
            assert_eq!(&before, expected);
        }
        assert!(decode_next(&mut r, &before).is_err(), "there's no more");
    }

    #[test]
    fn what_comes_off_the_network_is_checked() {
        // An element below zero, and a number that never ends
        let mut w = Writer::new();
        put_number(&mut w, 1 << 1);
        put_signed(&mut w, -5);
        put_signed(&mut w, 0);
        assert!(decode_next(&mut Reader::new(w.as_slice()), &DescriptorCopy::default()).is_err());
        let endless = [0xFFu8; 12];
        assert!(decode_next(&mut Reader::new(&endless), &DescriptorCopy::default()).is_err());
    }
}
