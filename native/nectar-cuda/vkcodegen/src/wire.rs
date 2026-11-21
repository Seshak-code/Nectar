//! The tables vkwire's interpreter reads (vkwire/src/generated.rs), the lender's call thunks,
//! and the borrower driver's entry points (vk/src/generated.rs).

use std::collections::BTreeMap;
use std::fmt::Write;

use crate::registry::{Decl, Registry, Struct, Type};
use crate::types::{Known, c_type, ident, rust_type};

/// Extensions Nectar doesn't carry: their handles or memory only mean something on the
/// borrower (file descriptors, host pointers, displays), or they call back into the program
pub const NOT_CARRIED: &[&str] = &[
    "VK_KHR_external_memory_fd",
    "VK_KHR_external_semaphore_fd",
    "VK_KHR_external_fence_fd",
    "VK_EXT_external_memory_host",
    "VK_EXT_external_memory_dma_buf",
    "VK_EXT_image_drm_format_modifier",
    "VK_EXT_debug_report",
    "VK_EXT_debug_utils",
    "VK_EXT_debug_marker",
    "VK_KHR_display",
    "VK_KHR_display_swapchain",
    "VK_EXT_direct_mode_display",
    "VK_EXT_display_surface_counter",
    "VK_EXT_display_control",
    "VK_EXT_acquire_drm_display",
    "VK_KHR_surface",
    "VK_KHR_swapchain",
    "VK_KHR_get_surface_capabilities2",
    "VK_EXT_headless_surface",
    "VK_KHR_surface_protected_capabilities",
    "VK_EXT_surface_maintenance1",
    "VK_EXT_swapchain_maintenance1",
    "VK_KHR_present_id",
    "VK_KHR_present_wait",
    "VK_KHR_present_id2",
    "VK_KHR_present_wait2",
    "VK_KHR_incremental_present",
    "VK_KHR_shared_presentable_image",
    "VK_EXT_hdr_metadata",
    "VK_GOOGLE_display_timing",
    "VK_KHR_device_group_creation",
    "VK_EXT_device_fault",
    "VK_LUNARG_direct_driver_loading",
    "VK_EXT_layer_settings",
    "VK_KHR_portability_enumeration",
];

/// Written by hand in the borrower's driver (vk/src/lib.rs) and the lender (server/src/vulkan.rs)
pub const OVERRIDES: &[&str] = &[
    "vkGetInstanceProcAddr",
    "vkGetDeviceProcAddr",
    "vkCreateInstance",
    "vkDestroyInstance",
    "vkEnumerateInstanceVersion",
    "vkEnumerateInstanceExtensionProperties",
    "vkEnumerateInstanceLayerProperties",
    "vkEnumerateDeviceLayerProperties",
    "vkEnumeratePhysicalDevices",
    "vkEnumerateDeviceExtensionProperties",
    "vkCreateDevice",
    "vkDestroyDevice",
    "vkMapMemory",
    "vkMapMemory2",
    "vkUnmapMemory",
    "vkUnmapMemory2",
    "vkFlushMappedMemoryRanges",
    "vkInvalidateMappedMemoryRanges",
    "vkFreeMemory",
];

const WAITS: &[&str] = &[
    "vkWaitForFences",
    "vkWaitSemaphores",
    "vkQueueWaitIdle",
    "vkDeviceWaitIdle",
    "vkGetQueryPoolResults",
];

/// Creates whose failure a program checks for: these wait for the lender's answer
const SYNC_CREATES: &[&str] = &[
    "vkAllocateMemory",
    "vkCreateGraphicsPipelines",
    "vkCreateComputePipelines",
    "vkCreateRayTracingPipelinesKHR",
    "vkCreateRayTracingPipelinesNV",
    "vkAllocateDescriptorSets",
    "vkCreatePipelineBinariesKHR",
    "vkCreateShadersEXT",
];

pub struct Wire<'a> {
    registry: &'a Registry,
    known: &'a Known<'a>,
    /// Struct name to its position in TYPES
    index: BTreeMap<String, usize>,
    structs: Vec<(String, &'a Struct)>,
    /// Length expressions written as functions
    functions: String,
    next_function: usize,
}

impl<'a> Wire<'a> {
    pub fn new(registry: &'a Registry, known: &'a Known<'a>) -> Self {
        let mut structs = Vec::new();
        for name in &registry.required_types {
            if let Some(Type::Struct(s)) = registry.types.get(name)
                && !s.union
                && known.is(name)
            {
                structs.push((name.clone(), s));
            }
        }
        let index = structs.iter().enumerate().map(|(i, (n, _))| (n.clone(), i)).collect();
        Self { registry, known, index, structs, functions: String::new(), next_function: 0 }
    }

    fn struct_index(&self, name: &str) -> Option<usize> {
        self.index.get(self.registry.resolve(name)).copied()
    }

    fn is_union(&self, name: &str) -> bool {
        matches!(self.registry.types.get(self.registry.resolve(name)), Some(Type::Struct(s)) if s.union)
    }

    fn handle(&self, name: &str) -> Option<bool> {
        match self.registry.types.get(self.registry.resolve(name)) {
            Some(Type::Handle { dispatchable }) => Some(*dispatchable),
            _ => None,
        }
    }

    fn rust_scalar(&self, name: &str) -> String {
        c_type(name).map(str::to_string).unwrap_or_else(|| name.to_string())
    }

    /// The element behind the innermost pointer level
    fn element(&self, ty: &str, has_len: bool) -> String {
        if ty == "void" {
            return if has_len { "Elem::Raw(1)".into() } else { "Elem::Unsupported(\"untyped pointer\")".into() };
        }
        if ty == "size_t" {
            return "Elem::SizeT".into();
        }
        if let Some(dispatchable) = self.handle(ty) {
            return format!("Elem::Handle({dispatchable})");
        }
        if let Some(index) = self.struct_index(ty) {
            return format!("Elem::Struct({index})");
        }
        if matches!(self.registry.types.get(self.registry.resolve(ty)), Some(Type::FuncPointer)) {
            return "Elem::Unsupported(\"function pointer\")".into();
        }
        format!("Elem::Raw(size_of::<{}>())", self.rust_scalar(ty))
    }

    /// A pointer member or parameter, level by level from the outermost
    fn pointer(&mut self, decl: &Decl, lens: &[String], level: usize) -> String {
        let levels = decl.pointers.len();
        let target_const = decl.pointers[levels - 1 - level];
        let len = lens.get(level).cloned().unwrap_or_else(|| "Len::One".into());
        let elem = if level + 1 == levels {
            self.element(&decl.ty, lens.get(level).is_some_and(|l| l != "Len::One"))
        } else {
            format!("Elem::Ptr({})", self.pointer(decl, lens, level + 1))
        };
        let out = !target_const && level == 0;
        format!("&Pointer {{ elem: {elem}, len: {len}, out: {out}, inout: false }}")
    }

    /// An expression from the registry (altlen) as Rust, its names read by `read`
    fn expression(&self, text: &str, read: impl Fn(&str) -> Option<String>) -> Option<String> {
        let mut out = String::new();
        let mut word = String::new();
        let flush = |word: &mut String, out: &mut String| -> Option<()> {
            if word.is_empty() {
                return Some(());
            }
            if word.chars().all(|c| c.is_ascii_digit()) {
                *out += word;
            } else if let Some(value) = self.registry.constant(word) {
                if let crate::registry::Value::Int(v) = value {
                    *out += &v.to_string();
                }
            } else {
                *out += &read(word)?;
            }
            word.clear();
            Some(())
        };
        for c in text.chars() {
            if c.is_ascii_alphanumeric() || c == '_' {
                word.push(c);
            } else {
                flush(&mut word, &mut out)?;
                out.push(c);
            }
        }
        flush(&mut word, &mut out)?;
        Some(out)
    }

    fn new_function(&mut self, signature: &str, body: &str) -> String {
        let name = format!("len_{}", self.next_function);
        self.next_function += 1;
        let _ = writeln!(self.functions, "fn {name}{signature} {{\n    unsafe {{ ({body}) as usize }}\n}}");
        name
    }

    /// The length rules of a member's pointer levels
    fn member_lens(&mut self, owner: &str, s: &Struct, decl: &Decl, member_index: &BTreeMap<String, usize>) -> Option<Vec<String>> {
        let Some(len) = &decl.len else { return Some(Vec::new()) };
        let mut lens = Vec::new();
        for part in len.split(',') {
            let part = part.trim();
            lens.push(if part == "null-terminated" {
                "Len::NullTerminated".into()
            } else if part == "1" {
                "Len::One".into()
            } else if let Some(&index) = member_index.get(part) {
                format!("Len::Sibling({index})")
            } else if part.starts_with("latexmath") {
                return None;
            } else {
                let names: Vec<String> = s.members.iter().map(|m| m.name.clone()).collect();
                let body = self.expression(part, |word| {
                    names.contains(&word.to_string()).then(|| format!("(*(p as *const {owner})).{} as usize", ident(word)))
                })?;
                let function = self.new_function("(p: *const u8) -> usize", &body);
                format!("Len::StructExpr({function})")
            });
        }
        Some(lens)
    }

    fn param_lens(&mut self, params: &[Decl], decl: &Decl) -> Option<Vec<String>> {
        let Some(len) = &decl.len else { return Some(Vec::new()) };
        let mut lens = Vec::new();
        for part in len.split(',') {
            let part = part.trim();
            let position = |name: &str| params.iter().position(|p| p.name == name);
            lens.push(if part == "null-terminated" {
                "Len::NullTerminated".into()
            } else if part == "1" {
                "Len::One".into()
            } else if let Some(index) = position(part) {
                if params[index].pointers.is_empty() {
                    format!("Len::Sibling({index})")
                } else {
                    format!("Len::SiblingDeref({index})")
                }
            } else if let Some((param, field)) = part.split_once("->") {
                let index = position(param)?;
                let owner = &params[index].ty;
                let field_type = match self.registry.types.get(self.registry.resolve(owner)) {
                    Some(Type::Struct(s)) => s.members.iter().find(|m| m.name == field)?.ty.clone(),
                    _ => return None,
                };
                format!(
                    "Len::SiblingField({index}, std::mem::offset_of!({owner}, {}), size_of::<{}>() as u8)",
                    ident(field),
                    self.rust_scalar(&field_type)
                )
            } else if part.starts_with("latexmath") {
                return None;
            } else {
                let body = self.expression(part, |word| {
                    let index = params.iter().position(|p| p.name == word)?;
                    let ty = rust_type(self.registry, &params[index]);
                    Some(format!("(*(a[{index}] as *const {ty})) as usize"))
                })?;
                let function = self.new_function("(a: &[*const c_void]) -> usize", &body);
                format!("Len::ParamExpr({function})")
            });
        }
        Some(lens)
    }

    /// A member or parameter's Field
    fn field(&mut self, owner: &str, decl: &Decl, lens: Option<Vec<String>>, member: bool) -> String {
        if member && decl.name == "sType" && decl.ty == "VkStructureType" {
            return "Field::SType".into();
        }
        if member && decl.name == "pNext" {
            return "Field::Next".into();
        }
        if !member && decl.ty == "VkAllocationCallbacks" {
            // Never sent: the lender uses its own allocator
            return "Field::Raw(0)".into();
        }
        if !decl.pointers.is_empty() {
            let Some(lens) = lens else {
                return "Field::Unsupported(\"length the registry gives only as a formula\")".into();
            };
            return format!("Field::Ptr({})", self.pointer(decl, &lens, 0));
        }
        let count: usize = decl
            .array
            .iter()
            .map(|size| crate::types::array_size(self.registry, size))
            .product();
        if !member && !decl.array.is_empty() {
            // A C array parameter is a pointer to its first element
            let elem = self.element(&decl.ty, true);
            return format!("Field::Ptr(&Pointer {{ elem: {elem}, len: Len::Fixed({count}), out: false, inout: false }})");
        }
        if let Some(dispatchable) = self.handle(&decl.ty) {
            return if decl.array.is_empty() {
                format!("Field::Handle({dispatchable})")
            } else {
                format!("Field::HandleArray({dispatchable}, {count})")
            };
        }
        if !self.is_union(&decl.ty)
            && let Some(index) = self.struct_index(&decl.ty)
        {
            return if decl.array.is_empty() {
                format!("Field::Struct({index})")
            } else {
                format!("Field::StructArray({index}, {count})")
            };
        }
        if decl.ty == "size_t" && decl.array.is_empty() {
            return "Field::SizeT".into();
        }
        if matches!(self.registry.types.get(self.registry.resolve(&decl.ty)), Some(Type::FuncPointer)) {
            return "Field::Unsupported(\"function pointer\")".into();
        }
        let _ = owner;
        format!("Field::Raw(size_of::<{}>())", rust_type(self.registry, decl))
    }

    fn stype_value(&self, s: &Struct) -> String {
        s.members
            .iter()
            .find(|m| m.name == "sType")
            .and_then(|m| m.values.clone())
            .filter(|value| {
                self.registry
                    .enums
                    .get("VkStructureType")
                    .is_some_and(|group| group.values.iter().any(|(n, _)| n == value))
            })
            .unwrap_or_else(|| "0".into())
    }

    /// vkwire/src/generated.rs
    pub fn emit_tables(&mut self) -> (String, Vec<String>) {
        let mut types = String::from("pub static TYPES: &[TypeInfo] = &[\n");
        let mut by_stype = String::from("pub fn by_stype(stype: i32) -> Option<Index> {\n    match stype {\n");
        let structs = self.structs.clone();
        for (position, (name, s)) in structs.iter().enumerate() {
            let mut members = String::new();
            let mut member_index = BTreeMap::new();
            let mut emitted = 0;
            let mut bits: Option<(Vec<String>, u32)> = None;
            for decl in &s.members {
                if decl.bitfield.is_some() {
                    continue;
                }
                member_index.insert(decl.name.clone(), emitted);
                emitted += 1;
            }
            let mut emitted = 0;
            for decl in &s.members {
                if let Some(width) = decl.bitfield {
                    let (names, used) = bits.get_or_insert_with(|| (Vec::new(), 0));
                    names.push(decl.name.clone());
                    *used += width;
                    if *used >= 32 {
                        let field = names.join("_and_");
                        let _ = writeln!(
                            members,
                            "        Member {{ name: \"{field}\", offset: std::mem::offset_of!({name}, {field}), field: Field::Raw(4) }},"
                        );
                        bits = None;
                    }
                    continue;
                }
                let lens = if decl.pointers.is_empty() { Some(Vec::new()) } else { self.member_lens(name, s, decl, &member_index) };
                let field = self.field(name, decl, lens, true);
                let _ = writeln!(
                    members,
                    "        Member {{ name: \"{}\", offset: std::mem::offset_of!({name}, {}), field: {field} }},",
                    decl.name,
                    ident(&decl.name)
                );
                emitted += 1;
            }
            let _ = emitted;
            let stype = self.stype_value(s);
            let _ = writeln!(
                types,
                "    TypeInfo {{ name: \"{name}\", size: size_of::<{name}>(), stype: {stype}, members: &[\n{members}    ] }}, // {position}"
            );
            if stype != "0" {
                let _ = writeln!(by_stype, "        {stype} => Some({position}),");
            }
        }
        types += "];\n";
        by_stype += "        _ => None,\n    }\n}\n";

        let mut commands = String::from("pub static COMMANDS: &[CommandInfo] = &[\n");
        let mut names = Vec::new();
        for name in self.registry.required_commands.clone() {
            if !self.known.command(&name) {
                continue;
            }
            let command = &self.registry.commands[&name];
            let mut params = String::new();
            let mut unsupported: Option<String> = None;
            for decl in &command.params {
                let lens = if decl.pointers.is_empty() { Some(Vec::new()) } else { self.param_lens(&command.params, decl) };
                let field = self.field("", decl, lens, false);
                if field.contains("Unsupported") && unsupported.is_none() {
                    unsupported = Some(format!("parameter {}", decl.name));
                }
                let _ = writeln!(params, "        Param {{ name: \"{}\", field: {field} }},", decl.name);
            }
            // Counts read back into arrays the call fills are inputs too
            for decl in &command.params {
                if let Some(len) = &decl.len
                    && let Some(count) = len.split(',').next()
                    && command.params.iter().any(|p| p.name == count && !p.pointers.is_empty())
                {
                    let from = format!("Param {{ name: \"{count}\", field: Field::Ptr(&Pointer {{");
                    if let Some(at) = params.find(&from) {
                        let end = at + params[at..].find('\n').unwrap();
                        let line = params[at..end].replace("inout: false", "inout: true");
                        params.replace_range(at..end, &line);
                    }
                }
            }
            let first = command.params.first().map(|p| self.registry.resolve(&p.ty).to_string()).unwrap_or_default();
            let level = match first.as_str() {
                "VkInstance" | "VkPhysicalDevice" => "Level::Instance",
                "VkDevice" | "VkQueue" | "VkCommandBuffer" => "Level::Device",
                _ => "Level::Global",
            };
            let ret = match command.ret.as_str() {
                "void" => "Ret::Void",
                "VkResult" => "Ret::Result",
                "uint32_t" | "VkBool32" => "Ret::U32",
                "PFN_vkVoidFunction" => "Ret::PFN",
                _ => "Ret::U64",
            };
            let client_handles = name.starts_with("vkCreate") || name.starts_with("vkAllocate");
            let has_outputs = params.contains("out: true");
            let only_handle_outputs = command.params.iter().all(|p| {
                p.pointers.is_empty() || p.pointers[p.pointers.len() - 1] || self.handle(&p.ty).is_some()
            });
            // A result other than VK_SUCCESS that still means success (VK_NOT_READY from
            // vkGetFenceStatus) is an answer the program needs
            let only_success = command.success.iter().all(|code| code == "VK_SUCCESS");
            let mode = if WAITS.contains(&name.as_str()) {
                "Mode::Wait"
            } else if matches!(ret, "Ret::Void" | "Ret::Result")
                && only_success
                && (!has_outputs || (client_handles && only_handle_outputs))
                && !SYNC_CREATES.contains(&name.as_str())
            {
                "Mode::Async"
            } else {
                "Mode::Sync"
            };
            // Commands only extensions Nectar doesn't carry bring: the loader answers them itself
            // (debug callbacks), or they belong to the borrower's own screen
            let not_carried = !self.registry.core_commands.contains_key(&name)
                && self.registry.extensions.iter().any(|e| e.commands.contains(&name))
                && self.registry.extensions.iter().filter(|e| e.commands.contains(&name)).all(|e| NOT_CARRIED.contains(&e.name.as_str()));
            let unsupported = match (unsupported, not_carried) {
                (_, true) => "Some(\"its extension isn't carried\")".to_string(),
                (Some(reason), _) => format!("Some(\"{reason}\")"),
                (None, _) => "None".into(),
            };
            let _ = writeln!(
                commands,
                "    CommandInfo {{ name: \"{name}\", id: function_id(\"{name}\"), params: &[\n{params}    ], ret: {ret}, level: {level}, mode: {mode}, client_handles: {client_handles}, unsupported: {unsupported} }},"
            );
            names.push(name);
        }
        commands += "];\n";

        let mut out = String::from(
            "// Generated by nectar-vk-codegen from vk.xml; do not edit.\n\
             #![allow(non_snake_case, clippy::all, unused_unsafe, unused_imports)]\n\n\
             use std::ffi::{c_char, c_int, c_void};\n\n\
             use nectar_cuda_protocol::function_id;\n\
             use nectar_vk_sys::*;\n\n\
             use crate::tables::*;\n\n",
        );
        out += &types;
        out += "\n";
        out += &by_stype;
        out += "\n";
        out += &commands;
        out += "\n";
        out += &self.functions;
        out += &self.extensions();
        out += &self.aliases(&names);
        (out, names)
    }

    /// Extensions whose commands can all be carried
    fn extensions(&self) -> String {
        let mut out = String::from("\n/// Extensions whose commands can all be carried, with their spec versions\npub static EXTENSIONS: &[(&str, u32)] = &[\n");
        for extension in &self.registry.extensions {
            if NOT_CARRIED.contains(&extension.name.as_str()) {
                continue;
            }
            let commands_ok = extension.commands.iter().all(|c| {
                let target = self.registry.command_aliases.get(c).cloned().unwrap_or(c.clone());
                self.known.command(&target)
            });
            let types_ok = extension.types.iter().all(|t| self.known.is(t));
            if commands_ok && types_ok {
                let _ = writeln!(out, "    (\"{}\", {}),", extension.name, extension.spec_version);
            }
        }
        out + "];\n"
    }

    fn aliases(&self, names: &[String]) -> String {
        let mut out = String::from("\n/// A command's own name, for one of its aliases (such as a KHR name of a core command)\npub fn command_name(name: &str) -> &str {\n    match name {\n");
        for (alias, target) in &self.registry.command_aliases {
            if names.contains(target) {
                let _ = writeln!(out, "        \"{alias}\" => \"{target}\",");
            }
        }
        out + "        other => other,\n    }\n}\n"
    }

    /// The lender's call thunks: a command called with arguments held in u64 slots
    pub fn emit_thunks(&self, names: &[String]) -> String {
        let mut out = String::from(
            "// Generated by nectar-vk-codegen from vk.xml; do not edit.\n\
             #![allow(non_snake_case, clippy::all)]\n\n\
             use std::ffi::c_void;\n\n\
             use nectar_vk_sys::*;\n\n\
             use crate::slots::{FromSlot, ToSlot};\n\n\
             /// Calls COMMANDS[index] at `function` with arguments from `a`\n\
             pub unsafe fn call(index: usize, function: *const c_void, a: &[u64]) -> u64 {\n    unsafe {\n        match index {\n",
        );
        for (index, name) in names.iter().enumerate() {
            let command = &self.registry.commands[name];
            let arguments: Vec<String> = (0..command.params.len()).map(|i| format!("FromSlot::from_slot(a[{i}])")).collect();
            let _ = writeln!(
                out,
                "            {index} => std::mem::transmute::<*const c_void, PFN_{name}>(function)({}).to_slot(),",
                arguments.join(", ")
            );
        }
        out += "            _ => unreachable!(),\n        }\n    }\n}\n";
        out
    }

    /// The borrower driver's entry points: each passes its arguments to crate::call
    pub fn emit_entries(&self, names: &[String]) -> String {
        let mut out = String::from(
            "// Generated by nectar-vk-codegen from vk.xml; do not edit.\n\
             #![allow(non_snake_case, clippy::all, unused_variables, unused_imports)]\n\n\
             use std::ffi::{c_char, c_int, c_void};\n\n\
             use nectar_vk_sys::*;\n\
             use nectar_vk_wire::slots::FromSlot;\n\n",
        );
        let mut lookup = String::from(
            "/// A generated entry point by command name\npub fn entry(name: &str) -> PFN_vkVoidFunction {\n    unsafe {\n        match name {\n",
        );
        for (index, name) in names.iter().enumerate() {
            let command = &self.registry.commands[name];
            let params: Vec<String> = command
                .params
                .iter()
                .map(|p| format!("{}: {}", ident(&p.name), rust_type(self.registry, p)))
                .collect();
            let arguments: Vec<String> = command
                .params
                .iter()
                .map(|p| format!("&{} as *const _ as *const c_void", ident(&p.name)))
                .collect();
            let (ret, convert) = match command.ret.as_str() {
                "void" => (String::new(), ";".to_string()),
                other => (
                    format!(" -> {}", c_type(other).unwrap_or(other)),
                    String::new(),
                ),
            };
            let call = format!("crate::call({index}, &[{}])", arguments.join(", "));
            let body = if convert == ";" { format!("{call};") } else { format!("FromSlot::from_slot({call})") };
            let _ = writeln!(
                out,
                "pub unsafe extern \"system\" fn {name}({}){ret} {{\n    unsafe {{ {body} }}\n}}",
                params.join(", ")
            );
            let _ = writeln!(
                lookup,
                "            \"{name}\" => std::mem::transmute::<PFN_{name}, PFN_vkVoidFunction>({name}),"
            );
        }
        lookup += "            _ => None,\n        }\n    }\n}\n";
        out + "\n" + &lookup
    }
}
