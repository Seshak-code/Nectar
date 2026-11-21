//! Generates Nectar's CUDA code from cuda.h and the real driver's export list.
//!
//! Usage: nectar-cuda-codegen <CUDA include dir> [--refresh-exports]
//!
//! Needs libclang (set LIBCLANG_PATH). Writes, relative to the workspace:
//!   sys/src/bindings.rs        CUDA types and constants
//!   protocol/src/functions.rs  every exported function, for protocol ids
//!   client/src/generated.rs    the borrower's exports
//!   server/src/generated.rs    the lender's dispatch
//!   trace/src/generated.rs     the tracer's pass-through exports
//!   codegen/report.md          how each export is handled, and why not when it isn't
//!
//! --refresh-exports first rereads codegen/nvcuda-exports.txt from System32\nvcuda.dll.
//! Regenerating after a new CUDA release is: point this at the new headers, rebuild, test.

mod emit;
mod model;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use model::{Function, Mode, Types};
use syn::{ForeignItem, Item};

/// Written by hand on both sides (client/src/lib.rs, server/src/main.rs)
pub const OVERRIDES: &[&str] = &[
    "cuInit",
    "cuGetProcAddress",
    "cuGetProcAddress_v2",
    "cuGetExportTable",
    "cuGetErrorName",
    "cuGetErrorString",
    "cuDeviceGetName",
    "cuMemcpyHtoD_v2",
    "cuMemcpyHtoD_v2_ptds",
    "cuMemcpyDtoH_v2",
    "cuMemcpyDtoH_v2_ptds",
    "cuDeviceGetLuid",
    "cuModuleLoadData",
    "cuModuleLoadDataEx",
    "cuLibraryLoadData",
    "cuLaunchKernel",
    "cuLaunchKernel_ptsz",
    "cuMemcpyHtoDAsync_v2",
    "cuMemcpyHtoDAsync_v2_ptsz",
    "cuMemcpyDtoHAsync_v2",
    "cuMemcpyDtoHAsync_v2_ptsz",
    "cuMemGetAllocationGranularity",
    "cuMemCreate",
    "cuMemSetAccess",
    "cuDeviceGetPCIBusId",
    // Host memory stays on the borrower; these never reach the lender
    "cuMemHostAlloc",
    "cuMemAllocHost_v2",
    "cuMemFreeHost",
    "cuMemHostGetFlags",
    "cuMemHostRegister_v2",
    "cuMemHostUnregister",
];

/// How one export of the real driver is handled
pub enum Plan {
    /// Marshalled by generated code
    Generated(Function, Vec<Mode>),
    /// Written by hand
    Override(Function),
    /// Exported with its signature, reports CUDA_ERROR_NOT_SUPPORTED
    Unsupported(Function, String),
    /// Not declared in cuda.h: exported without parameters, reports CUDA_ERROR_NOT_SUPPORTED
    Undeclared,
}

fn read_exports(driver: &Path) -> Vec<String> {
    use object::read::pe::PeFile64;

    let data = fs::read(driver).unwrap_or_else(|error| panic!("cannot read {}: {error}", driver.display()));
    let file = PeFile64::parse(&*data).expect("not a 64-bit PE file");
    let mut names: Vec<String> = file
        .export_table()
        .expect("bad export table")
        .expect("no export table")
        .exports()
        .expect("bad exports")
        .into_iter()
        .filter_map(|export| export.name.map(|name| String::from_utf8_lossy(name).into_owned()))
        .collect();
    names.sort();
    names
}

/// The symbol cuGetProcAddress is asked for: the name without its version and stream suffixes
fn base_symbol(export: &str) -> &str {
    let mut name = export;
    for suffix in ["_ptds", "_ptsz"] {
        name = name.strip_suffix(suffix).unwrap_or(name);
    }
    if let Some(index) = name.rfind("_v")
        && name[index + 2..].chars().all(|c| c.is_ascii_digit())
        && index + 2 < name.len()
    {
        name = &name[..index];
    }
    name
}

/// The declaration for an export: its own, or the one its stream variant shares
fn signature<'a>(export: &str, declared: &'a BTreeMap<String, syn::Signature>) -> Option<&'a syn::Signature> {
    if let Some(signature) = declared.get(export) {
        return Some(signature);
    }
    for suffix in ["_ptds", "_ptsz"] {
        if let Some(base) = export.strip_suffix(suffix)
            && let Some(signature) = declared.get(base)
        {
            return Some(signature);
        }
        if let Some(signature) = declared.get(&format!("{export}{suffix}")) {
            return Some(signature);
        }
    }
    None
}

fn write(root: &Path, relative: &str, contents: &str) {
    let path = root.join(relative);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, contents).unwrap_or_else(|error| panic!("cannot write {}: {error}", path.display()));
    println!("wrote {relative}");
}

fn main() {
    let mut include = None;
    let mut refresh = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--refresh-exports" => refresh = true,
            _ => include = Some(PathBuf::from(arg)),
        }
    }
    let include = include.expect("usage: nectar-cuda-codegen <CUDA include dir> [--refresh-exports]");
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();

    let exports_file = root.join("codegen/nvcuda-exports.txt");
    if refresh {
        let system = std::env::var_os("SystemRoot").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
        let exports = read_exports(&system.join(r"System32\nvcuda.dll"));
        fs::write(&exports_file, exports.join("\n") + "\n").unwrap();
        println!("read {} exports from the driver", exports.len());
    }
    let exports: Vec<String> =
        fs::read_to_string(&exports_file).expect("no export list; run with --refresh-exports").lines().map(str::to_string).collect();

    // __CUDA_API_VERSION_INTERNAL declares every variant: versioned, legacy and per-thread stream
    let header = include.join("cuda.h");
    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .clang_args(["-x", "c", "-D__CUDA_API_VERSION_INTERNAL", &format!("-I{}", include.display())])
        .allowlist_item("(cu|CU|cuuint).*")
        .default_enum_style(bindgen::EnumVariation::Consts)
        .prepend_enum_name(false)
        .generate_comments(false)
        .layout_tests(false)
        .derive_default(true)
        .generate()
        .expect("bindgen failed")
        .to_string();

    let file = syn::parse_file(&bindings).expect("bindgen output does not parse");
    let mut type_items = Vec::new();
    let mut declared = BTreeMap::new();
    let mut versions = Vec::new();
    for item in file.items {
        match item {
            Item::ForeignMod(foreign) => {
                for item in foreign.items {
                    if let ForeignItem::Fn(function) = item {
                        declared.insert(function.sig.ident.to_string(), function.sig);
                    }
                }
            }
            Item::Type(alias) if alias.ident.to_string().starts_with("PFN_") => {
                if let Some(version) = model::Version::from_alias(&alias) {
                    versions.push(version);
                }
            }
            other => type_items.push(other),
        }
    }

    let types = Types::new(&type_items);
    let (proc_addresses, unmatched) = model::proc_addresses(&types, &versions, &exports, &declared);
    let version = fs::read_to_string(&header)
        .unwrap()
        .lines()
        .find_map(|line| line.strip_prefix("#define CUDA_VERSION ").map(|value| value.trim().to_string()))
        .unwrap_or_default();

    let plans: Vec<(String, Plan)> = exports
        .iter()
        .map(|export| {
            let plan = match signature(export, &declared) {
                None => Plan::Undeclared,
                Some(signature) => {
                    let mut function = Function::from_signature(signature);
                    function.name = export.clone();
                    if OVERRIDES.contains(&export.as_str()) {
                        Plan::Override(function)
                    } else {
                        match model::plan(&types, &function) {
                            Ok(modes) => Plan::Generated(function, modes),
                            Err(reason) => Plan::Unsupported(function, reason),
                        }
                    }
                }
            };
            (export.clone(), plan)
        })
        .collect();

    let header_note = format!("// Generated by nectar-cuda-codegen from cuda.h (CUDA_VERSION {version}). Do not edit.\n\n");

    let bindings_file = syn::File { shebang: None, attrs: Vec::new(), items: type_items };
    write(&root, "sys/src/bindings.rs", &(header_note.clone() + &prettyplease::unparse(&bindings_file)));

    let mut bases: Vec<&str> = exports.iter().map(|export| base_symbol(export)).collect();
    bases.sort();
    bases.dedup();
    write(&root, "protocol/src/functions.rs", &(header_note.clone() + &emit::protocol(&exports, &bases, &plans)));
    write(&root, "client/src/generated.rs", &(header_note.clone() + &emit::client(&plans)));
    write(&root, "server/src/generated.rs", &(header_note.clone() + &emit::server(&plans)));
    write(&root, "trace/src/generated.rs", &(header_note + &emit::trace(&plans)));

    // The report is what to review after regenerating
    let mut report = format!(
        "# Nectar CUDA coverage\n\nGenerated from cuda.h (CUDA_VERSION {version}) and {} driver exports.\n\n",
        exports.len()
    );
    let count = |test: fn(&Plan) -> bool| plans.iter().filter(|(_, plan)| test(plan)).count();
    let _ = writeln!(report, "| Handling | Exports |\n| --- | --- |");
    let _ = writeln!(report, "| Generated | {} |", count(|plan| matches!(plan, Plan::Generated(..))));
    let _ = writeln!(report, "| Written by hand | {} |", count(|plan| matches!(plan, Plan::Override(..))));
    let _ = writeln!(report, "| Not supported yet | {} |", count(|plan| matches!(plan, Plan::Unsupported(..))));
    let _ = writeln!(report, "| Not in cuda.h (driver internals) | {} |", count(|plan| matches!(plan, Plan::Undeclared)));
    let _ = writeln!(
        report,
        "\ncuGetProcAddress can hand out {} (symbol, version) variants.",
        proc_addresses.len()
    );
    if !unmatched.is_empty() {
        let _ = writeln!(report, "\n## Versions in cudaTypedefs.h with no matching export\n");
        for version in &unmatched {
            let _ = writeln!(report, "- `{version}`");
        }
    }
    let _ = writeln!(report, "\n## Not supported yet\n");
    for (name, plan) in &plans {
        if let Plan::Unsupported(_, reason) = plan {
            let _ = writeln!(report, "- `{name}`: {reason}");
        }
    }
    let _ = writeln!(report, "\n## Not in cuda.h\n");
    for (name, plan) in &plans {
        if let Plan::Undeclared = plan {
            let _ = writeln!(report, "- `{name}`");
        }
    }
    write(&root, "codegen/report.md", &report);
}
