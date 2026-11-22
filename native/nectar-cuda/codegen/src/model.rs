//! Classifies the CUDA driver API's parameters into what can be carried over the network.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use syn::{Fields, FnArg, Item, Pat, Type, UseTree};

/// What a C type is, for marshalling
#[derive(Clone, Debug)]
pub enum Kind {
    /// Integers, floats, enums, device pointers
    Scalar,
    Char,
    Void,
    /// A pointer to an opaque struct, such as CUcontext: the lender's value, passed through
    Handle,
    /// A struct or union by value
    Record(String),
    Callback,
    Pointer {
        mutable: bool,
        target: Box<Kind>,
    },
    Array(Box<Kind>),
    Other(String),
}

/// How one parameter crosses the network
#[derive(Clone, Debug)]
pub enum Mode {
    /// By value, as its raw bytes
    Value,
    /// A non-const pointer to one plain value: copied to the lender and back, NULL allowed
    InOut(Type),
    /// A const pointer to one plain record: copied to the lender, NULL allowed
    In(Type),
    /// A const char* string, NULL allowed
    Str,
}

pub struct Param {
    pub name: String,
    pub ty: Type,
}

pub struct Function {
    pub name: String,
    pub params: Vec<Param>,
}

impl Function {
    pub fn from_signature(signature: &syn::Signature) -> Self {
        let params = signature
            .inputs
            .iter()
            .enumerate()
            .map(|(index, input)| match input {
                FnArg::Typed(typed) => {
                    let name = match &*typed.pat {
                        Pat::Ident(ident) => ident.ident.to_string(),
                        _ => format!("arg{index}"),
                    };
                    Param {
                        name,
                        ty: (*typed.ty).clone(),
                    }
                }
                FnArg::Receiver(_) => panic!("unexpected receiver in {}", signature.ident),
            })
            .collect();

        Self {
            name: signature.ident.to_string(),
            params,
        }
    }
}

pub struct Types {
    aliases: HashMap<String, Type>,
    /// Field types of every struct and union
    records: HashMap<String, Vec<Type>>,
    opaque: HashSet<String>,
    plain: RefCell<HashMap<String, bool>>,
}

impl Types {
    pub fn new(items: &[Item]) -> Self {
        let mut types = Types {
            aliases: HashMap::new(),
            records: HashMap::new(),
            opaque: HashSet::new(),
            plain: RefCell::new(HashMap::new()),
        };

        for item in items {
            match item {
                Item::Type(alias) => {
                    types
                        .aliases
                        .insert(alias.ident.to_string(), (*alias.ty).clone());
                }
                Item::Use(item_use) => {
                    // pub use self::cudaError_enum as CUresult;
                    if let UseTree::Path(path) = &item_use.tree
                        && let UseTree::Rename(rename) = &*path.tree
                    {
                        let target: Type = syn::parse_str(&rename.ident.to_string()).unwrap();
                        types.aliases.insert(rename.rename.to_string(), target);
                    }
                }
                Item::Struct(record) => {
                    let name = record.ident.to_string();
                    let fields: Vec<Type> = match &record.fields {
                        Fields::Named(named) => {
                            named.named.iter().map(|field| field.ty.clone()).collect()
                        }
                        Fields::Unnamed(unnamed) => unnamed
                            .unnamed
                            .iter()
                            .map(|field| field.ty.clone())
                            .collect(),
                        Fields::Unit => Vec::new(),
                    };
                    let is_opaque = match &record.fields {
                        Fields::Named(named) => {
                            named.named.len() == 1
                                && named.named[0]
                                    .ident
                                    .as_ref()
                                    .is_some_and(|ident| ident == "_unused")
                        }
                        _ => false,
                    };
                    if is_opaque {
                        types.opaque.insert(name.clone());
                    }
                    types.records.insert(name, fields);
                }
                Item::Union(record) => {
                    let fields = record
                        .fields
                        .named
                        .iter()
                        .map(|field| field.ty.clone())
                        .collect();
                    types.records.insert(record.ident.to_string(), fields);
                }
                _ => {}
            }
        }

        types
    }

    pub fn kind(&self, ty: &Type) -> Kind {
        match ty {
            Type::Ptr(pointer) => {
                let target = self.kind(&pointer.elem);
                if let Kind::Record(name) = &target
                    && self.opaque.contains(name)
                {
                    return Kind::Handle;
                }
                Kind::Pointer {
                    mutable: pointer.mutability.is_some(),
                    target: Box::new(target),
                }
            }
            Type::Array(array) => Kind::Array(Box::new(self.kind(&array.elem))),
            Type::BareFn(_) => Kind::Callback,
            Type::Paren(paren) => self.kind(&paren.elem),
            Type::Path(path) => {
                let ident = path.path.segments.last().unwrap().ident.to_string();
                match ident.as_str() {
                    "c_void" => Kind::Void,
                    "c_char" | "c_schar" | "c_uchar" => Kind::Char,
                    "c_short"
                    | "c_ushort"
                    | "c_int"
                    | "c_uint"
                    | "c_long"
                    | "c_ulong"
                    | "c_longlong"
                    | "c_ulonglong"
                    | "c_float"
                    | "c_double"
                    | "u8"
                    | "i8"
                    | "u16"
                    | "i16"
                    | "u32"
                    | "i32"
                    | "u64"
                    | "i64"
                    | "u128"
                    | "i128"
                    | "usize"
                    | "isize"
                    | "f32"
                    | "f64"
                    | "__BindgenBitfieldUnit" => Kind::Scalar,
                    "Option" => Kind::Callback,
                    _ => {
                        if let Some(alias) = self.aliases.get(&ident) {
                            self.kind(alias)
                        } else if self.records.contains_key(&ident) {
                            Kind::Record(ident)
                        } else {
                            Kind::Other(ident)
                        }
                    }
                }
            }
            _ => Kind::Other(quote::quote!(#ty).to_string()),
        }
    }

    /// Whether a record's bytes mean the same on the lender: no host pointers or callbacks.
    /// Handles inside are fine, since they are the lender's values.
    pub fn plain(&self, name: &str) -> bool {
        if let Some(&known) = self.plain.borrow().get(name) {
            return known;
        }
        if self.opaque.contains(name) {
            return false;
        }

        // A record that refers back to itself is not plain
        self.plain.borrow_mut().insert(name.to_string(), false);
        let plain = self.records.get(name).is_some_and(|fields| {
            fields
                .iter()
                .all(|field| self.plain_kind(&self.kind(field)))
        });
        self.plain.borrow_mut().insert(name.to_string(), plain);
        plain
    }

    fn plain_kind(&self, kind: &Kind) -> bool {
        match kind {
            Kind::Scalar | Kind::Char | Kind::Handle => true,
            Kind::Record(name) => self.plain(name),
            Kind::Array(element) => self.plain_kind(element),
            _ => false,
        }
    }

    /// How a parameter crosses the network, or why it cannot yet
    pub fn mode(&self, ty: &Type) -> Result<Mode, String> {
        match self.kind(ty) {
            Kind::Scalar | Kind::Char | Kind::Handle => Ok(Mode::Value),
            Kind::Record(name) if self.plain(&name) => Ok(Mode::Value),
            Kind::Record(name) => Err(format!("{name} by value holds host pointers")),
            Kind::Callback => Err("callback".into()),
            Kind::Pointer { mutable, target } => {
                let Type::Ptr(pointer) = ty else {
                    unreachable!()
                };
                let target_type = (*pointer.elem).clone();
                match *target {
                    Kind::Char if !mutable => Ok(Mode::Str),
                    Kind::Char => Err("output string".into()),
                    Kind::Void => Err("host memory".into()),
                    Kind::Scalar | Kind::Handle if mutable => Ok(Mode::InOut(target_type)),
                    Kind::Scalar | Kind::Handle => Err("input array".into()),
                    Kind::Record(name) if self.plain(&name) => Ok(if mutable {
                        Mode::InOut(target_type)
                    } else {
                        Mode::In(target_type)
                    }),
                    Kind::Record(name) => Err(format!("{name} holds host pointers")),
                    Kind::Pointer { .. } => Err("pointer to pointer".into()),
                    Kind::Callback => Err("callback".into()),
                    Kind::Other(name) => Err(format!("pointer to unknown type {name}")),
                    Kind::Array(_) => Err("pointer to array".into()),
                }
            }
            Kind::Void => Err("void by value".into()),
            Kind::Array(_) => Err("array by value".into()),
            Kind::Other(name) => Err(format!("unknown type {name}")),
        }
    }
}

/// A parameter that counts elements, such as numNodes or nbResources
fn counts(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    name == "count"
        || name.ends_with("Count")
        || lower.starts_with("num")
        || lower.starts_with("nb")
}

/// Plans a whole function. A pointer next to a count is probably an array, which a
/// single-value copy would overrun, so such functions wait for a hand-written version.
pub fn plan(types: &Types, function: &Function) -> Result<Vec<Mode>, String> {
    let modes = function
        .params
        .iter()
        .map(|param| {
            types
                .mode(&param.ty)
                .map_err(|reason| format!("{}: {reason}", param.name))
        })
        .collect::<Result<Vec<_>, _>>()?;

    for (index, param) in function.params.iter().enumerate() {
        if !counts(&param.name) {
            continue;
        }
        for (other, mode) in modes.iter().enumerate() {
            if other != index && matches!(mode, Mode::InOut(_) | Mode::In(_)) {
                return Err(format!(
                    "{}: may be an array counted by {}",
                    function.params[other].name, param.name
                ));
            }
        }
    }

    Ok(modes)
}

impl Types {
    /// A type with every alias resolved, for comparing signatures
    pub fn canonical(&self, ty: &Type) -> String {
        match ty {
            Type::Ptr(pointer) => {
                let mutability = if pointer.mutability.is_some() {
                    "mut"
                } else {
                    "const"
                };
                format!("*{mutability} {}", self.canonical(&pointer.elem))
            }
            Type::Array(array) => format!("[{}]", self.canonical(&array.elem)),
            Type::Path(path) => {
                let ident = path.path.segments.last().unwrap().ident.to_string();
                match self.aliases.get(&ident) {
                    Some(alias) => self.canonical(alias),
                    None => ident,
                }
            }
            other => quote::quote!(#other).to_string(),
        }
    }
}

/// One PFN_<symbol>_v<version>[_ptds|_ptsz] type from cudaTypedefs.h: a variant of a
/// function and the CUDA version that introduced it
pub struct Version {
    pub symbol: String,
    pub version: u32,
    pub per_thread: bool,
    pub params: Vec<Type>,
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let stream = if self.per_thread {
            " (per-thread stream)"
        } else {
            ""
        };
        write!(f, "{} {}{stream}", self.symbol, self.version)
    }
}

impl Version {
    pub fn from_alias(alias: &syn::ItemType) -> Option<Self> {
        let name = alias.ident.to_string();
        let mut rest = name.strip_prefix("PFN_")?;
        let mut per_thread = false;
        for suffix in ["_ptds", "_ptsz"] {
            if let Some(stripped) = rest.strip_suffix(suffix) {
                rest = stripped;
                per_thread = true;
            }
        }
        let index = rest.rfind("_v")?;
        let version = rest[index + 2..].parse().ok()?;

        // Option<unsafe extern "C" fn(...) -> CUresult>
        let Type::Path(path) = &*alias.ty else {
            return None;
        };
        let syn::PathArguments::AngleBracketed(arguments) = &path.path.segments.last()?.arguments
        else {
            return None;
        };
        let Some(syn::GenericArgument::Type(Type::BareFn(function))) = arguments.args.first()
        else {
            return None;
        };

        Some(Self {
            symbol: rest[..index].to_string(),
            version,
            per_thread,
            params: function
                .inputs
                .iter()
                .map(|input| input.ty.clone())
                .collect(),
        })
    }
}

/// What cuGetProcAddress hands out: for each symbol and version in cudaTypedefs.h, the export
/// with the same signature. Variants with the same signature go in version order, so
/// cuCtxCreate at 2000 is cuCtxCreate and at 3020 is cuCtxCreate_v2.
///
/// Returns (symbol, version, per-thread stream, export), sorted, and the versions no
/// export matched.
pub fn proc_addresses(
    types: &Types,
    versions: &[Version],
    exports: &[String],
    declared: &std::collections::BTreeMap<String, syn::Signature>,
) -> (Vec<(String, u32, bool, String)>, Vec<String>) {
    // Exports by (symbol, per-thread stream), in suffix order: none, _v2, _v3...
    let mut candidates: HashMap<(String, bool), Vec<(u32, String, Vec<String>)>> = HashMap::new();
    for export in exports {
        let Some(signature) = crate::signature(export, declared) else {
            continue;
        };
        let per_thread = export.ends_with("_ptds") || export.ends_with("_ptsz");
        let unsuffixed = export.trim_end_matches("_ptds").trim_end_matches("_ptsz");
        let order = unsuffixed
            .rfind("_v")
            .and_then(|index| unsuffixed[index + 2..].parse().ok())
            .unwrap_or(1);
        let params = signature
            .inputs
            .iter()
            .map(|input| match input {
                FnArg::Typed(typed) => types.canonical(&typed.ty),
                FnArg::Receiver(_) => String::new(),
            })
            .collect();
        candidates
            .entry((crate::base_symbol(export).to_string(), per_thread))
            .or_default()
            .push((order, export.clone(), params));
    }
    for list in candidates.values_mut() {
        list.sort();
    }

    let mut ordered: Vec<&Version> = versions.iter().collect();
    ordered.sort_by_key(|version| (version.symbol.clone(), version.per_thread, version.version));

    let mut table = Vec::new();
    let mut unmatched = Vec::new();
    let mut used: HashSet<String> = HashSet::new();
    for version in ordered {
        let params: Vec<String> = version
            .params
            .iter()
            .map(|ty| types.canonical(ty))
            .collect();
        // A newer version with an unchanged signature can keep the same export
        let found = candidates
            .get(&(version.symbol.clone(), version.per_thread))
            .and_then(|list| {
                list.iter()
                    .find(|(_, export, candidate)| !used.contains(export) && *candidate == params)
                    .or_else(|| {
                        list.iter()
                            .rev()
                            .find(|(_, _, candidate)| *candidate == params)
                    })
            });
        match found {
            Some((_, export, _)) => {
                used.insert(export.clone());
                table.push((
                    version.symbol.clone(),
                    version.version,
                    version.per_thread,
                    export.clone(),
                ));
            }
            None => unmatched.push(version.to_string()),
        }
    }

    table.sort();
    (table, unmatched)
}
