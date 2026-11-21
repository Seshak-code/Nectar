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
    Pointer { mutable: bool, target: Box<Kind> },
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
                    Param { name, ty: (*typed.ty).clone() }
                }
                FnArg::Receiver(_) => panic!("unexpected receiver in {}", signature.ident),
            })
            .collect();

        Self { name: signature.ident.to_string(), params }
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
                    types.aliases.insert(alias.ident.to_string(), (*alias.ty).clone());
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
                        Fields::Named(named) => named.named.iter().map(|field| field.ty.clone()).collect(),
                        Fields::Unnamed(unnamed) => unnamed.unnamed.iter().map(|field| field.ty.clone()).collect(),
                        Fields::Unit => Vec::new(),
                    };
                    let is_opaque = match &record.fields {
                        Fields::Named(named) => {
                            named.named.len() == 1 && named.named[0].ident.as_ref().is_some_and(|ident| ident == "_unused")
                        }
                        _ => false,
                    };
                    if is_opaque {
                        types.opaque.insert(name.clone());
                    }
                    types.records.insert(name, fields);
                }
                Item::Union(record) => {
                    let fields = record.fields.named.iter().map(|field| field.ty.clone()).collect();
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
                Kind::Pointer { mutable: pointer.mutability.is_some(), target: Box::new(target) }
            }
            Type::Array(array) => Kind::Array(Box::new(self.kind(&array.elem))),
            Type::BareFn(_) => Kind::Callback,
            Type::Paren(paren) => self.kind(&paren.elem),
            Type::Path(path) => {
                let ident = path.path.segments.last().unwrap().ident.to_string();
                match ident.as_str() {
                    "c_void" => Kind::Void,
                    "c_char" | "c_schar" | "c_uchar" => Kind::Char,
                    "c_short" | "c_ushort" | "c_int" | "c_uint" | "c_long" | "c_ulong" | "c_longlong"
                    | "c_ulonglong" | "c_float" | "c_double" | "u8" | "i8" | "u16" | "i16" | "u32" | "i32" | "u64"
                    | "i64" | "u128" | "i128" | "usize" | "isize" | "f32" | "f64" | "__BindgenBitfieldUnit" => {
                        Kind::Scalar
                    }
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
        let plain = self.records.get(name).is_some_and(|fields| fields.iter().all(|field| self.plain_kind(&self.kind(field))));
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
                let Type::Ptr(pointer) = ty else { unreachable!() };
                let target_type = (*pointer.elem).clone();
                match *target {
                    Kind::Char if !mutable => Ok(Mode::Str),
                    Kind::Char => Err("output string".into()),
                    Kind::Void => Err("host memory".into()),
                    Kind::Scalar | Kind::Handle if mutable => Ok(Mode::InOut(target_type)),
                    Kind::Scalar | Kind::Handle => Err("input array".into()),
                    Kind::Record(name) if self.plain(&name) => {
                        Ok(if mutable { Mode::InOut(target_type) } else { Mode::In(target_type) })
                    }
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
    name == "count" || name.ends_with("Count") || lower.starts_with("num") || lower.starts_with("nb")
}

/// Plans a whole function. A pointer next to a count is probably an array, which a
/// single-value copy would overrun, so such functions wait for a hand-written version.
pub fn plan(types: &Types, function: &Function) -> Result<Vec<Mode>, String> {
    let modes = function
        .params
        .iter()
        .map(|param| types.mode(&param.ty).map_err(|reason| format!("{}: {reason}", param.name)))
        .collect::<Result<Vec<_>, _>>()?;

    for (index, param) in function.params.iter().enumerate() {
        if !counts(&param.name) {
            continue;
        }
        for (other, mode) in modes.iter().enumerate() {
            if other != index && matches!(mode, Mode::InOut(_) | Mode::In(_)) {
                return Err(format!("{}: may be an array counted by {}", function.params[other].name, param.name));
            }
        }
    }

    Ok(modes)
}
