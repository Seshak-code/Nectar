//! Writes the generated Rust sources.

use std::fmt::Write as _;

use quote::quote;
use syn::Type;

use crate::Plan;
use crate::model::{Function, Mode};

fn ty(ty: &Type) -> String {
    quote!(#ty).to_string()
}

/// `a0: T0, a1: T1`
fn params(function: &Function) -> String {
    function
        .params
        .iter()
        .enumerate()
        .map(|(index, param)| format!("a{index}: {}", ty(&param.ty)))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `unsafe extern "system" fn(T0, T1) -> CUresult`
fn pointer_type(function: &Function) -> String {
    let types: Vec<String> = function.params.iter().map(|param| ty(&param.ty)).collect();
    format!(
        "unsafe extern \"system\" fn({}) -> CUresult",
        types.join(", ")
    )
}

fn arguments(function: &Function) -> String {
    (0..function.params.len())
        .map(|index| format!("a{index}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `/// cuMemAlloc_v2(dptr, bytesize)`
fn doc(function: &Function) -> String {
    let names: Vec<&str> = function
        .params
        .iter()
        .map(|param| param.name.as_str())
        .collect();
    format!("/// {}({})\n", function.name, names.join(", "))
}

fn hash(text: &str) -> u32 {
    text.bytes().fold(0x811c_9dc5u32, |hash, byte| {
        (hash ^ byte as u32).wrapping_mul(0x0100_0193)
    })
}

pub fn protocol(
    exports: &[String],
    proc_addresses: &[(String, u32, bool, String)],
    plans: &[(String, Plan)],
) -> String {
    let mut out = String::new();

    // Any change to what crosses the network changes this, so mismatched builds refuse to talk
    let mut signatures = String::new();
    for (name, plan) in plans {
        if let Plan::Generated(function, _) | Plan::Override(function) = plan {
            let _ = write!(signatures, "{name}({});", pointer_type(function));
        }
    }

    let _ = writeln!(
        out,
        "/// Identifies this generation of the protocol. Client and server must match."
    );
    let _ = writeln!(
        out,
        "pub const GENERATION: u32 = {:#010x};\n",
        hash(&signatures)
    );
    let _ = writeln!(out, "functions!(");
    for export in exports {
        let _ = writeln!(out, "    {export},");
    }
    let _ = writeln!(out, ");\n");
    let _ = writeln!(
        out,
        "/// What cuGetProcAddress hands out, from cudaTypedefs.h: (symbol, CUDA version that\n\
         /// introduced the variant, per-thread default stream, export), sorted"
    );
    let _ = writeln!(
        out,
        "pub const PROC_ADDRESSES: &[(&str, u32, bool, &str)] = &["
    );
    for (symbol, version, per_thread, export) in proc_addresses {
        let _ = writeln!(
            out,
            "    (\"{symbol}\", {version}, {per_thread}, \"{export}\"),"
        );
    }
    let _ = writeln!(out, "];");
    out
}

pub fn client(plans: &[(String, Plan)]) -> String {
    let mut out = String::from(
        "#![allow(unused_variables, clippy::all)]\n\n\
         use std::collections::HashMap;\n\
         use std::ffi::c_void;\n\
         use std::sync::OnceLock;\n\n\
         use nectar_cuda_protocol::{Writer, id};\n\
         use nectar_cuda_sys::*;\n\n\
         use crate::{call, finish, unsupported};\n\n",
    );

    for (name, plan) in plans {
        match plan {
            Plan::Generated(function, modes) => {
                let mut request = String::new();
                let mut reply = String::new();
                for (index, mode) in modes.iter().enumerate() {
                    match mode {
                        Mode::Value => {
                            let _ = writeln!(request, "    request.value(&a{index});");
                        }
                        Mode::InOut(_) => {
                            let _ = writeln!(
                                request,
                                "    unsafe {{ request.optional(a{index}.cast_const()) }};"
                            );
                            let _ = writeln!(
                                reply,
                                "        unsafe {{ reply.optional_into(a{index})? }};"
                            );
                        }
                        Mode::In(_) => {
                            let _ =
                                writeln!(request, "    unsafe {{ request.optional(a{index}) }};");
                        }
                        Mode::Str => {
                            let _ = writeln!(request, "    unsafe {{ request.string(a{index}) }};");
                        }
                    }
                }

                let binding = if modes.is_empty() {
                    "let request"
                } else {
                    "let mut request"
                };
                let _ = write!(
                    out,
                    "{doc}#[unsafe(no_mangle)]\npub unsafe extern \"system\" fn {name}({params}) -> CUresult {{\n    \
                     {binding} = Writer::new();\n{request}    \
                     finish(call(id::{name}, &[request.as_slice()]), |reply| {{\n{reply}        Ok(())\n    }})\n}}\n\n",
                    doc = doc(function),
                    params = params(function),
                );
            }
            Plan::Unsupported(function, reason) => {
                let _ = write!(
                    out,
                    "{doc}/// Not supported yet: {reason}\n#[unsafe(no_mangle)]\npub unsafe extern \"system\" fn {name}({params}) -> CUresult {{\n    \
                     unsupported(id::{name})\n}}\n\n",
                    doc = doc(function),
                    params = params(function),
                );
            }
            Plan::Undeclared => {
                // Returning without reading arguments is safe in the x64 calling convention
                let _ = write!(
                    out,
                    "/// Not declared in cuda.h\n#[unsafe(no_mangle)]\npub unsafe extern \"system\" fn {name}() -> CUresult {{\n    \
                     unsupported(id::{name})\n}}\n\n"
                );
            }
            Plan::Override(_) => {}
        }
    }

    let _ = writeln!(out, "/// Every export by name, for cuGetProcAddress");
    let _ = writeln!(
        out,
        "pub(crate) fn exports() -> &'static HashMap<&'static str, *const c_void> {{"
    );
    let _ = writeln!(
        out,
        "    struct Table(HashMap<&'static str, *const c_void>);"
    );
    let _ = writeln!(out, "    unsafe impl Send for Table {{}}");
    let _ = writeln!(out, "    unsafe impl Sync for Table {{}}");
    let _ = writeln!(out, "    static TABLE: OnceLock<Table> = OnceLock::new();");
    let _ = writeln!(out, "    &TABLE.get_or_init(|| Table(HashMap::from([");
    for (name, plan) in plans {
        let path = if matches!(plan, Plan::Override(_)) {
            format!("crate::{name}")
        } else {
            name.clone()
        };
        let _ = writeln!(out, "        (\"{name}\", {path} as *const c_void),");
    }
    let _ = writeln!(out, "    ]))).0\n}}");
    out
}

/// Checks the server makes before certain calls. With unified addressing, the driver takes
/// an address that is not GPU memory to be host memory, which on the lender would be the
/// server's own memory, so copies between arbitrary addresses only go ahead for GPU memory.
fn server_check(name: &str) -> Option<&'static str> {
    Some(match name {
        "cuMemcpy" | "cuMemcpy_ptds" | "cuMemcpyAsync" | "cuMemcpyAsync_ptsz" => {
            "if !crate::gpu_memory(&[a0, a1]) { return Ok(Some(CUDA_ERROR_INVALID_VALUE)); }"
        }
        _ => return None,
    })
}

/// Bookkeeping the server does after certain calls succeed, so it can release what a
/// borrower leaves behind
fn server_hook(name: &str) -> Option<&'static str> {
    Some(match name {
        "cuMemAlloc_v2" | "cuMemAllocPitch_v2" => {
            "if let Some(pointer) = a0.get() { session.allocated(pointer); }"
        }
        "cuMemFree_v2" => "session.freed(a0);",
        "cuDevicePrimaryCtxRetain" => {
            "if let Some(context) = a0.get() { session.retained(a1, context); }"
        }
        "cuDevicePrimaryCtxRelease_v2" => "session.released(a0);",
        "cuMemAddressReserve" => {
            "if let Some(pointer) = a0.get() { session.reserved(pointer, a1); }"
        }
        "cuMemAddressFree" => "session.unreserved(a0);",
        "cuMemRelease" => "session.released_memory(a0);",
        // Launches take kernels and functions alike; the server must know which to ask about
        "cuLibraryGetKernel" => {
            "if let Some(kernel) = a0.get() { session.kernels.insert(kernel as usize); }"
        }
        _ => return None,
    })
}

pub fn server(plans: &[(String, Plan)]) -> String {
    let mut out = String::from(
        "#![allow(unused_mut, clippy::all)]\n\n\
         use std::ffi::c_void;\n\
         use std::io;\n\n\
         use nectar_cuda_protocol::{Reader, Writer, id};\n\
         use nectar_cuda_sys::*;\n\n\
         use crate::{Session, Slot};\n\n\
         /// Carries out a generated call. None when the function is not generated here or the\n\
         /// driver lacks it.\n\
         pub(crate) unsafe fn dispatch(\n    address: *const c_void,\n    session: &mut Session,\n    function: u32,\n    \
         request: &mut Reader,\n    reply: &mut Writer,\n) -> io::Result<Option<CUresult>> {\n    let result = match function {\n",
    );

    for (name, plan) in plans {
        let Plan::Generated(function, modes) = plan else {
            continue;
        };

        let mut body = String::new();
        let mut write_back = String::new();
        for (index, mode) in modes.iter().enumerate() {
            match mode {
                Mode::Value => {
                    let _ = writeln!(
                        body,
                        "            let a{index}: {} = unsafe {{ request.value()? }};",
                        ty(&function.params[index].ty)
                    );
                }
                Mode::InOut(target) => {
                    let _ = writeln!(
                        body,
                        "            let mut a{index} = Slot::<{}>::read(request)?;",
                        ty(target)
                    );
                    let _ = writeln!(write_back, "            a{index}.write(reply);");
                }
                Mode::In(target) => {
                    let _ = writeln!(
                        body,
                        "            let mut a{index} = Slot::<{}>::read(request)?;",
                        ty(target)
                    );
                }
                Mode::Str => {
                    let _ = writeln!(body, "            let a{index} = request.string()?;");
                }
            }
        }

        let arguments: Vec<String> = modes
            .iter()
            .enumerate()
            .map(|(index, mode)| match mode {
                Mode::Value => format!("a{index}"),
                Mode::InOut(_) => format!("a{index}.as_mut_ptr()"),
                Mode::In(_) => format!("a{index}.as_ptr()"),
                Mode::Str => {
                    format!("a{index}.as_ref().map_or(std::ptr::null(), |text| text.as_ptr())")
                }
            })
            .collect();

        let hook = server_hook(name)
            .map(|hook| format!("            if result == CUDA_SUCCESS {{ {hook} }}\n"))
            .unwrap_or_default();
        let check = server_check(name)
            .map(|check| format!("            {check}\n"))
            .unwrap_or_default();

        let _ = write!(
            out,
            "        id::{name} => {{\n            \
             let call: {pointer} = unsafe {{ std::mem::transmute(address) }};\n{body}{check}            \
             let result = unsafe {{ call({arguments}) }};\n{hook}{write_back}            result\n        }}\n",
            pointer = pointer_type(function),
            arguments = arguments.join(", "),
        );
    }

    out.push_str("        _ => return Ok(None),\n    };\n    Ok(Some(result))\n}\n");
    out
}

/// Hand-written in the tracer, since they hand out function pointers
const TRACE_OVERRIDES: &[&str] = &[
    "cuGetProcAddress",
    "cuGetProcAddress_v2",
    "cuGetExportTable",
];

pub fn trace(plans: &[(String, Plan)]) -> String {
    let mut out = String::from(
        "#![allow(clippy::all)]\n\n\
         use std::collections::HashMap;\n\
         use std::ffi::c_void;\n\
         use std::sync::OnceLock;\n\
         use std::time::Instant;\n\n\
         use nectar_cuda_sys::*;\n\n\
         use crate::{real, record};\n\n",
    );

    let mut traced = Vec::new();
    for (index, (name, plan)) in plans.iter().enumerate() {
        let function = match plan {
            Plan::Generated(function, _)
            | Plan::Override(function)
            | Plan::Unsupported(function, _) => function,
            Plan::Undeclared => continue,
        };
        traced.push(name.as_str());
        if TRACE_OVERRIDES.contains(&name.as_str()) {
            continue;
        }

        let _ = write!(
            out,
            "{doc}#[unsafe(no_mangle)]\npub unsafe extern \"system\" fn {name}({params}) -> CUresult {{\n    \
             let Some(address) = real({index}) else {{ return CUDA_ERROR_NOT_SUPPORTED }};\n    \
             let call: {pointer} = unsafe {{ std::mem::transmute(address) }};\n    \
             let started = Instant::now();\n    \
             let result = unsafe {{ call({arguments}) }};\n    \
             record({index}, started, result);\n    result\n}}\n\n",
            doc = doc(function),
            params = params(function),
            pointer = pointer_type(function),
            arguments = arguments(function),
        );
    }

    let _ = writeln!(out, "/// The traced exports by name, for cuGetProcAddress");
    let _ = writeln!(
        out,
        "pub(crate) fn exports() -> &'static HashMap<&'static str, *const c_void> {{"
    );
    let _ = writeln!(
        out,
        "    struct Table(HashMap<&'static str, *const c_void>);"
    );
    let _ = writeln!(out, "    unsafe impl Send for Table {{}}");
    let _ = writeln!(out, "    unsafe impl Sync for Table {{}}");
    let _ = writeln!(out, "    static TABLE: OnceLock<Table> = OnceLock::new();");
    let _ = writeln!(out, "    &TABLE.get_or_init(|| Table(HashMap::from([");
    for name in traced {
        let path = if TRACE_OVERRIDES.contains(&name) {
            format!("crate::{name}")
        } else {
            name.to_string()
        };
        let _ = writeln!(out, "        (\"{name}\", {path} as *const c_void),");
    }
    let _ = writeln!(out, "    ]))).0\n}}");
    out
}
