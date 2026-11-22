//! The Vulkan registry (vk.xml) read into what the generators need: types, enumerants,
//! constants, commands, and which of them the included features and extensions require.
//!
//! Included: every feature for the `vulkan` API and every extension supported for it, except
//! provisional ones and those tied to a window system or OS (`platform`), which come with the
//! frames-on-screen work. Elements marked for Vulkan SC only are skipped.

use std::collections::{BTreeMap, BTreeSet};

use roxmltree::{Document, Node};

/// A C declaration of a struct member or command parameter
#[derive(Clone, Debug, Default)]
pub struct Decl {
    pub name: String,
    /// The type named, without pointers or arrays
    pub ty: String,
    /// `const` before the type: the innermost data is read-only
    pub inner_const: bool,
    /// Pointer levels, innermost first; true where that level's target is const
    pub pointers: Vec<bool>,
    /// Fixed array sizes, outermost first, as written (a number or a constant's name)
    pub array: Vec<String>,
    pub bitfield: Option<u32>,
    /// What gives an array's length: another member or parameter, `null-terminated`, an
    /// expression (from altlen when the registry has one), or several, comma-separated
    pub len: Option<String>,
    /// Per pointer level, outermost first
    pub optional: Vec<bool>,
    /// For sType members, the one value it must have
    pub values: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Struct {
    pub members: Vec<Decl>,
    pub union: bool,
    pub returned_only: bool,
    /// The structs this one may extend through pNext
    pub extends: Vec<String>,
}

#[derive(Clone, Debug)]
pub enum Type {
    /// A C basic type or one named in <types requires=…> (uint32_t, float…)
    External,
    /// typedef of another type, or an opaque struct (ANativeWindow)
    Base(Option<String>),
    Handle { dispatchable: bool },
    Enum,
    Bitmask { flags64: bool, bits: Option<String> },
    Struct(Struct),
    FuncPointer,
    Alias(String),
}

#[derive(Clone, Debug)]
pub struct Command {
    pub name: String,
    pub ret: String,
    pub params: Vec<Decl>,
    /// The results that mean success, such as VK_SUCCESS and VK_NOT_READY
    pub success: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Int(i64),
    U64(u64),
    Float(f32),
    Text(String),
}

#[derive(Clone, Debug, Default)]
pub struct EnumGroup {
    pub bitmask: bool,
    pub flags64: bool,
    /// Value names in order, with their values (aliases resolved)
    pub values: Vec<(String, i64)>,
}

#[derive(Clone, Debug)]
pub struct Extension {
    pub name: String,
    pub number: u32,
    /// instance or device
    pub kind: String,
    pub commands: Vec<String>,
    pub types: Vec<String>,
    pub spec_version: u32,
}

#[derive(Default)]
pub struct Registry {
    pub types: BTreeMap<String, Type>,
    pub enums: BTreeMap<String, EnumGroup>,
    pub constants: Vec<(String, Value)>,
    pub commands: BTreeMap<String, Command>,
    pub command_aliases: BTreeMap<String, String>,
    /// What the included features and extensions require
    pub required_types: BTreeSet<String>,
    pub required_commands: Vec<String>,
    /// Commands of each core version, such as "1.1"
    pub core_commands: BTreeMap<String, String>,
    pub extensions: Vec<Extension>,
}

/// Whether an element applies to the Vulkan API (rather than only to Vulkan SC)
fn for_vulkan(node: Node) -> bool {
    node.attribute("api")
        .is_none_or(|api| api.split(',').any(|api| api == "vulkan"))
}

/// The text of a node and its children, comments left out
fn text(node: Node) -> String {
    let mut out = String::new();
    for child in node.children() {
        if child.is_text() {
            out += child.text().unwrap_or_default();
        } else if child.is_element() && !child.has_tag_name("comment") {
            out += &text(child);
        }
    }
    out
}

/// A <member> or <param>, read as a C declaration
fn decl(node: Node) -> Decl {
    let mut before = String::new();
    let mut between = String::new();
    let mut after = String::new();
    let (mut ty, mut name) = (String::new(), String::new());
    for child in node.children() {
        if child.has_tag_name("type") {
            ty = text(child);
        } else if child.has_tag_name("name") {
            name = text(child);
        } else if child.has_tag_name("comment") {
        } else {
            let piece = if child.is_text() { child.text().unwrap_or_default().to_string() } else { text(child) };
            match (ty.is_empty(), name.is_empty()) {
                (true, _) => before += &piece,
                (false, true) => between += &piece,
                _ => after += &piece,
            }
        }
    }
    // "* const*": a pointer to const pointers. Level 0 is the innermost.
    let mut pointers = Vec::new();
    let stars: Vec<&str> = between.split('*').collect();
    for index in 0..stars.len().saturating_sub(1) {
        let target_const = if index == 0 {
            before.contains("const")
        } else {
            stars[index].contains("const")
        };
        pointers.push(target_const);
    }
    let mut array = Vec::new();
    let mut rest = after.trim();
    while let Some(open) = rest.find('[') {
        let close = rest[open..].find(']').map(|at| open + at).unwrap_or(rest.len());
        array.push(rest[open + 1..close].trim().to_string());
        rest = &rest[(close + 1).min(rest.len())..];
    }
    let bitfield = after
        .trim()
        .strip_prefix(':')
        .and_then(|bits| bits.trim().parse().ok());
    let len = node
        .attribute("altlen")
        .or(node.attribute("len"))
        .map(str::to_string);
    let optional = node
        .attribute("optional")
        .map(|value| value.split(',').map(|part| part == "true").collect())
        .unwrap_or_default();
    Decl {
        name,
        ty,
        inner_const: before.contains("const"),
        pointers,
        array,
        bitfield,
        len,
        optional,
        values: node.attribute("values").map(str::to_string),
    }
}

fn parse_value(text: &str) -> Value {
    let text = text.trim();
    match text {
        "(~0U)" => return Value::Int(u32::MAX as i64),
        "(~1U)" => return Value::Int((u32::MAX - 1) as i64),
        "(~2U)" => return Value::Int((u32::MAX - 2) as i64),
        "(~0ULL)" => return Value::U64(u64::MAX),
        _ => {}
    }
    if let Some(quoted) = text.strip_prefix('"').and_then(|t| t.strip_suffix('"')) {
        return Value::Text(quoted.to_string());
    }
    if let Some(float) = text.strip_suffix('F').or(text.strip_suffix('f'))
        && let Ok(value) = float.parse()
    {
        return Value::Float(value);
    }
    let digits = text.trim_end_matches(['U', 'L']);
    if let Some(hex) = digits.strip_prefix("0x") {
        return Value::Int(i64::from_str_radix(hex, 16).unwrap_or_default());
    }
    Value::Int(digits.parse().unwrap_or_default())
}

/// The value of an enumerant an extension or feature adds
fn extension_value(node: Node, extension_number: Option<u32>) -> Option<i64> {
    if let Some(offset) = node.attribute("offset") {
        let number: i64 = node
            .attribute("extnumber")
            .and_then(|n| n.parse().ok())
            .or(extension_number.map(i64::from))?;
        let value = 1_000_000_000 + (number - 1) * 1000 + offset.parse::<i64>().ok()?;
        return Some(if node.attribute("dir") == Some("-") { -value } else { value });
    }
    if let Some(bit) = node.attribute("bitpos") {
        return Some(1i64 << bit.parse::<u32>().ok()?);
    }
    match parse_value(node.attribute("value")?) {
        Value::Int(value) => Some(value),
        _ => None,
    }
}

impl Registry {
    pub fn parse(xml: &str) -> Self {
        let document = Document::parse(xml).expect("vk.xml is not valid XML");
        let root = document.root_element();
        let mut registry = Registry::default();
        let mut enum_aliases: Vec<(String, String, String)> = Vec::new();

        for types in root.children().filter(|n| n.has_tag_name("types")) {
            for node in types.children().filter(|n| n.has_tag_name("type") && for_vulkan(*n)) {
                registry.read_type(node);
            }
        }

        for group in root.children().filter(|n| n.has_tag_name("enums")) {
            let name = group.attribute("name").unwrap_or_default().to_string();
            if group.attribute("type") == Some("constants") {
                for value in group.children().filter(|n| n.has_tag_name("enum")) {
                    if let Some(text) = value.attribute("value") {
                        registry
                            .constants
                            .push((value.attribute("name").unwrap().to_string(), parse_value(text)));
                    }
                }
                continue;
            }
            let entry = registry.enums.entry(name.clone()).or_default();
            entry.bitmask = group.attribute("type") == Some("bitmask");
            entry.flags64 = group.attribute("bitwidth") == Some("64");
            for value in group.children().filter(|n| n.has_tag_name("enum") && for_vulkan(*n)) {
                let value_name = value.attribute("name").unwrap().to_string();
                if let Some(alias) = value.attribute("alias") {
                    enum_aliases.push((name.clone(), value_name, alias.to_string()));
                } else if let Some(number) = extension_value(value, None) {
                    entry.values.push((value_name, number));
                }
            }
        }

        for commands in root.children().filter(|n| n.has_tag_name("commands")) {
            for node in commands.children().filter(|n| n.has_tag_name("command") && for_vulkan(*n)) {
                if let Some(alias) = node.attribute("alias") {
                    registry
                        .command_aliases
                        .insert(node.attribute("name").unwrap().to_string(), alias.to_string());
                    continue;
                }
                let proto = node.children().find(|n| n.has_tag_name("proto")).unwrap();
                let proto = decl(proto);
                let params = node
                    .children()
                    .filter(|n| n.has_tag_name("param") && for_vulkan(*n))
                    .map(decl)
                    .collect();
                let ret = if proto.pointers.is_empty() { proto.ty.clone() } else { format!("{}*", proto.ty) };
                registry.commands.insert(
                    proto.name.clone(),
                    Command {
                        name: proto.name,
                        ret,
                        params,
                        success: node
                            .attribute("successcodes")
                            .map(|codes| codes.split(',').map(str::to_string).collect())
                            .unwrap_or_default(),
                    },
                );
            }
        }

        let mut required_commands = BTreeSet::new();
        for feature in root.children().filter(|n| n.has_tag_name("feature") && for_vulkan(*n)) {
            let version = feature.attribute("number").unwrap_or_default().to_string();
            for (kind, name) in registry.read_requires(feature, None, &mut enum_aliases) {
                if kind == "command" {
                    required_commands.insert(name.clone());
                    registry.core_commands.entry(name).or_insert_with(|| version.clone());
                }
            }
        }
        for extensions in root.children().filter(|n| n.has_tag_name("extensions")) {
            for node in extensions.children().filter(|n| n.has_tag_name("extension")) {
                let supported = node
                    .attribute("supported")
                    .is_some_and(|s| s.split(',').any(|api| api == "vulkan"));
                let number: u32 = node.attribute("number").and_then(|n| n.parse().ok()).unwrap_or(0);
                let included = supported
                    && node.attribute("platform").is_none()
                    && node.attribute("provisional") != Some("true");
                // Enumerants of extensions left out are still added, so values stay complete
                let required = registry.read_requires(node, Some(number), &mut enum_aliases);
                if !included {
                    continue;
                }
                let mut commands = Vec::new();
                let mut types = Vec::new();
                for (kind, name) in &required {
                    if kind == "command" {
                        required_commands.insert(name.clone());
                        commands.push(name.clone());
                    } else {
                        types.push(name.clone());
                    }
                }
                let spec_version = node
                    .descendants()
                    .filter(|n| n.has_tag_name("enum"))
                    .find(|n| n.attribute("name").is_some_and(|name| name.ends_with("_SPEC_VERSION")))
                    .and_then(|n| n.attribute("value"))
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1);
                registry.extensions.push(Extension {
                    name: node.attribute("name").unwrap().to_string(),
                    number,
                    kind: node.attribute("type").unwrap_or("device").to_string(),
                    commands,
                    types,
                    spec_version,
                });
            }
        }

        // Aliases of enumerants, now that every value is known
        for (group, name, alias) in enum_aliases {
            let entry = registry.enums.entry(group).or_default();
            if let Some(value) = entry.values.iter().find(|(n, _)| *n == alias).map(|(_, v)| *v)
                && !entry.values.iter().any(|(n, _)| *n == name)
            {
                entry.values.push((name, value));
            }
        }

        registry.required_commands = required_commands
            .into_iter()
            .filter(|name| !registry.command_aliases.contains_key(name))
            .collect();
        registry.close_types();
        registry
    }

    fn read_type(&mut self, node: Node) {
        let category = node.attribute("category").unwrap_or_default();
        // A funcpointer names itself inside <proto>
        let name = node
            .attribute("name")
            .map(str::to_string)
            .or_else(|| node.descendants().find(|n| n.has_tag_name("name")).map(text))
            .unwrap_or_default();
        if name.is_empty() {
            return;
        }
        if let Some(alias) = node.attribute("alias") {
            self.types.insert(name, Type::Alias(alias.to_string()));
            return;
        }
        let ty = match category {
            "" | "include" | "define" => {
                if category.is_empty() {
                    Type::External
                } else {
                    return;
                }
            }
            "basetype" => Type::Base(node.children().find(|n| n.has_tag_name("type")).map(text)),
            "handle" => Type::Handle {
                dispatchable: text(node).contains("VK_DEFINE_HANDLE"),
            },
            "enum" => Type::Enum,
            "bitmask" => Type::Bitmask {
                flags64: text(node).contains("VkFlags64"),
                bits: node
                    .attribute("requires")
                    .or(node.attribute("bitvalues"))
                    .map(str::to_string),
            },
            "struct" | "union" => Type::Struct(Struct {
                members: node
                    .children()
                    .filter(|n| n.has_tag_name("member") && for_vulkan(*n))
                    .map(decl)
                    .collect(),
                union: category == "union",
                returned_only: node.attribute("returnedonly") == Some("true"),
                extends: node
                    .attribute("structextends")
                    .map(|list| list.split(',').map(str::to_string).collect())
                    .unwrap_or_default(),
            }),
            "funcpointer" => Type::FuncPointer,
            _ => return,
        };
        self.types.insert(name, ty);
    }

    /// The commands and types a feature or extension requires, adding the enumerants it adds
    fn read_requires(
        &mut self,
        node: Node,
        extension_number: Option<u32>,
        aliases: &mut Vec<(String, String, String)>,
    ) -> Vec<(String, String)> {
        let mut required = Vec::new();
        for require in node.children().filter(|n| n.has_tag_name("require") && for_vulkan(*n)) {
            for item in require.children().filter(|n| n.is_element() && for_vulkan(*n)) {
                let Some(name) = item.attribute("name") else { continue };
                match item.tag_name().name() {
                    "command" => required.push(("command".to_string(), name.to_string())),
                    "type" => {
                        self.required_types.insert(name.to_string());
                        required.push(("type".to_string(), name.to_string()));
                    }
                    "enum" => {
                        if let Some(group) = item.attribute("extends") {
                            if let Some(alias) = item.attribute("alias") {
                                aliases.push((group.to_string(), name.to_string(), alias.to_string()));
                            } else if let Some(value) = extension_value(item, extension_number) {
                                let entry = self.enums.entry(group.to_string()).or_default();
                                if !entry.values.iter().any(|(n, _)| n == name) {
                                    entry.values.push((name.to_string(), value));
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        required
    }

    /// Adds every type the required commands and types use, through members, parameters,
    /// aliases and bitmask bits
    fn close_types(&mut self) {
        let mut pending: Vec<String> = self.required_types.iter().cloned().collect();
        for command in &self.required_commands {
            if let Some(command) = self.commands.get(command) {
                pending.push(command.ret.trim_end_matches('*').to_string());
                pending.extend(command.params.iter().map(|p| p.ty.clone()));
            }
        }
        let mut seen = BTreeSet::new();
        while let Some(name) = pending.pop() {
            if !seen.insert(name.clone()) {
                continue;
            }
            match self.types.get(&name) {
                Some(Type::Alias(target)) => pending.push(target.clone()),
                Some(Type::Struct(s)) => pending.extend(s.members.iter().map(|m| m.ty.clone())),
                Some(Type::Bitmask { bits: Some(bits), .. }) => pending.push(bits.clone()),
                Some(Type::Base(Some(base))) => pending.push(base.clone()),
                _ => {}
            }
        }
        self.required_types = seen;
    }

    /// A type's name with aliases followed
    pub fn resolve<'a>(&'a self, mut name: &'a str) -> &'a str {
        while let Some(Type::Alias(target)) = self.types.get(name) {
            name = target;
        }
        name
    }

    pub fn constant(&self, name: &str) -> Option<&Value> {
        self.constants.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }
}
