//! Generates Nectar's Vulkan code from the Vulkan registry.
//!
//! Usage: nectar-vk-codegen <folder with vk.xml>
//!
//! Writes, relative to the workspace:
//!   vksys/src/generated.rs     Vulkan types and command signatures
//!   vkwire/src/generated.rs    tables of every struct and command, for the wire interpreter
//!   vkwire/src/thunks.rs       the lender's calls into the real driver
//!   vk/src/generated.rs        the borrower driver's entry points
//!   vkcodegen/report.md        what is generated, and what is left out and why
//!
//! Regenerating after a new Vulkan release is: point this at the new registry, rebuild, test.

mod registry;
mod types;
mod wire;

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use registry::Registry;

fn write(root: &Path, relative: &str, contents: &str) {
    let path = root.join(relative);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, contents)
        .unwrap_or_else(|error| panic!("cannot write {}: {error}", path.display()));
}

fn main() {
    let folder = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("usage: nectar-vk-codegen <folder with vk.xml>"));
    let xml = fs::read_to_string(folder.join("vk.xml")).expect("cannot read vk.xml");
    let registry = Registry::parse(&xml);
    let known = types::Known::new(&registry);

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    write(root, "vksys/src/generated.rs", &types::emit(&registry, &known));
    let mut wire = wire::Wire::new(&registry, &known);
    let (tables, names) = wire.emit_tables();
    write(root, "vkwire/src/generated.rs", &tables);
    write(root, "vkwire/src/thunks.rs", &wire.emit_thunks(&names));
    write(root, "vk/src/generated.rs", &wire.emit_entries(&names));

    let declared = registry.required_commands.iter().filter(|c| known.command(c)).count();
    let mut report = String::from("# Vulkan code generated from vk.xml\n\n");
    let _ = writeln!(
        report,
        "- Types required: {}, declared: {}",
        registry.required_types.len(),
        registry.required_types.iter().filter(|t| known.is(t)).count()
    );
    let _ = writeln!(
        report,
        "- Commands required: {}, declared: {declared}",
        registry.required_commands.len()
    );
    let _ = writeln!(report, "- Extensions included: {}\n", registry.extensions.len());
    report += "## Commands left out\n\nThey use a type the registry doesn't define here (the video codec headers' StdVideo types).\n\n";
    for command in registry.required_commands.iter().filter(|c| !known.command(c)) {
        let _ = writeln!(report, "- `{command}`");
    }
    write(root, "vkcodegen/report.md", &report);
    println!(
        "{} commands declared of {}; see vkcodegen/report.md",
        declared,
        registry.required_commands.len()
    );
}
