//! A small SPIR-V writer: enough to write a compute shader in Rust, with no shader compiler to
//! install. Values are 32-bit unsigned or signed integers and floats. The shader is built when it's
//! needed, so there's no build step and nothing to ship.
//!
//! Each method emits one instruction and returns its result's id. `if_` and `while_` emit the
//! structured control flow Vulkan requires.

use std::collections::HashMap;

/// An id: a result, a variable or a function
pub type Id = u32;

const OP_EXT_INST_IMPORT: u32 = 11;
const OP_EXT_INST: u32 = 12;
const OP_MEMORY_MODEL: u32 = 14;
const OP_ENTRY_POINT: u32 = 15;
const OP_EXECUTION_MODE: u32 = 16;
const OP_CAPABILITY: u32 = 17;
const OP_TYPE_VOID: u32 = 19;
const OP_TYPE_BOOL: u32 = 20;
const OP_TYPE_INT: u32 = 21;
const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
const OP_TYPE_STRUCT: u32 = 30;
const OP_TYPE_POINTER: u32 = 32;
const OP_TYPE_FUNCTION: u32 = 33;
const OP_CONSTANT: u32 = 43;
const OP_CONSTANT_COMPOSITE: u32 = 44;
const OP_FUNCTION: u32 = 54;
const OP_FUNCTION_PARAMETER: u32 = 55;
const OP_FUNCTION_END: u32 = 56;
const OP_FUNCTION_CALL: u32 = 57;
const OP_VARIABLE: u32 = 59;
const OP_LOAD: u32 = 61;
const OP_STORE: u32 = 62;
const OP_ACCESS_CHAIN: u32 = 65;
const OP_DECORATE: u32 = 71;
const OP_MEMBER_DECORATE: u32 = 72;
const OP_COMPOSITE_EXTRACT: u32 = 81;
const OP_CONVERT_F_TO_S: u32 = 110;
const OP_CONVERT_U_TO_F: u32 = 112;
const OP_BITCAST: u32 = 124;
const OP_I_ADD: u32 = 128;
const OP_F_ADD: u32 = 129;
const OP_I_SUB: u32 = 130;
const OP_F_SUB: u32 = 131;
const OP_I_MUL: u32 = 132;
const OP_F_MUL: u32 = 133;
const OP_U_DIV: u32 = 134;
const OP_F_DIV: u32 = 136;
const OP_LOGICAL_AND: u32 = 167;
const OP_LOGICAL_NOT: u32 = 168;
const OP_I_EQUAL: u32 = 170;
const OP_SELECT: u32 = 169;
const OP_I_NOT_EQUAL: u32 = 171;
const OP_U_LESS_THAN: u32 = 176;
const OP_SHIFT_RIGHT_LOGICAL: u32 = 194;
const OP_SHIFT_RIGHT_ARITHMETIC: u32 = 195;
const OP_SHIFT_LEFT_LOGICAL: u32 = 196;
const OP_BITWISE_OR: u32 = 197;
const OP_BITWISE_XOR: u32 = 198;
const OP_BITWISE_AND: u32 = 199;
const OP_LOOP_MERGE: u32 = 246;
const OP_SELECTION_MERGE: u32 = 247;
const OP_LABEL: u32 = 248;
const OP_BRANCH: u32 = 249;
const OP_BRANCH_CONDITIONAL: u32 = 250;
const OP_RETURN: u32 = 253;
const OP_RETURN_VALUE: u32 = 254;

const STORAGE_INPUT: u32 = 1;
const STORAGE_UNIFORM: u32 = 2;
const STORAGE_PRIVATE: u32 = 6;
const STORAGE_FUNCTION: u32 = 7;
const STORAGE_PUSH_CONSTANT: u32 = 9;

const DECORATE_BLOCK: u32 = 2;
const DECORATE_BUFFER_BLOCK: u32 = 3;
const DECORATE_ARRAY_STRIDE: u32 = 6;
const DECORATE_BUILT_IN: u32 = 11;
const DECORATE_BINDING: u32 = 33;
const DECORATE_DESCRIPTOR_SET: u32 = 34;
const DECORATE_OFFSET: u32 = 35;
const BUILT_IN_GLOBAL_INVOCATION_ID: u32 = 28;

const GLSL_ROUND: u32 = 1;
const GLSL_U_MIN: u32 = 38;
const GLSL_S_CLAMP: u32 = 45;
const GLSL_FIND_I_LSB: u32 = 73;
const GLSL_FIND_U_MSB: u32 = 75;

fn instruction(words: &mut Vec<u32>, op: u32, operands: &[u32]) {
    words.push(((operands.len() as u32 + 1) << 16) | op);
    words.extend_from_slice(operands);
}

/// A storage buffer of u32s
#[derive(Clone, Copy)]
pub struct Buffer(Id);

/// A variable holding one value
#[derive(Clone, Copy)]
pub struct Variable {
    id: Id,
    of: Id,
}

/// An array in the invocation's private memory
#[derive(Clone, Copy)]
pub struct Table {
    id: Id,
    of: Id,
}

/// Push constants of 4 bytes each
pub struct Push {
    id: Id,
    members: Vec<Id>,
}

/// The shader being written
#[derive(Default)]
pub struct Shader {
    next: Id,
    glsl: Id,
    decorations: Vec<u32>,
    globals: Vec<u32>,
    functions: Vec<u32>,
    /// The current function's variables, which must open its first block
    locals: Vec<u32>,
    body: Vec<u32>,
    types: HashMap<Vec<u32>, Id>,
    constants: HashMap<(Id, u32), Id>,
    invocation_id: Id,
    entry: Id,
    pub void: Id,
    pub boolean: Id,
    pub uint: Id,
    pub int: Id,
    pub float: Id,
}

impl Shader {
    pub fn new() -> Self {
        let mut shader = Shader { next: 1, ..Shader::default() };
        shader.glsl = shader.id();
        shader.void = shader.type_of(OP_TYPE_VOID, &[]);
        shader.boolean = shader.type_of(OP_TYPE_BOOL, &[]);
        shader.uint = shader.type_of(OP_TYPE_INT, &[32, 0]);
        shader.int = shader.type_of(OP_TYPE_INT, &[32, 1]);
        shader.float = shader.type_of(OP_TYPE_FLOAT, &[32]);
        shader
    }

    fn id(&mut self) -> Id {
        let id = self.next;
        self.next += 1;
        id
    }

    fn type_of(&mut self, op: u32, operands: &[u32]) -> Id {
        let mut key = vec![op];
        key.extend_from_slice(operands);
        if let Some(&id) = self.types.get(&key) {
            return id;
        }
        let id = self.id();
        let mut words = vec![id];
        words.extend_from_slice(operands);
        instruction(&mut self.globals, op, &words);
        self.types.insert(key, id);
        id
    }

    fn pointer_to(&mut self, storage: u32, of: Id) -> Id {
        self.type_of(OP_TYPE_POINTER, &[storage, of])
    }

    fn constant(&mut self, of: Id, bits: u32) -> Id {
        if let Some(&id) = self.constants.get(&(of, bits)) {
            return id;
        }
        let id = self.id();
        instruction(&mut self.globals, OP_CONSTANT, &[of, id, bits]);
        self.constants.insert((of, bits), id);
        id
    }

    pub fn u(&mut self, value: u32) -> Id {
        self.constant(self.uint, value)
    }

    pub fn i(&mut self, value: i32) -> Id {
        self.constant(self.int, value as u32)
    }

    pub fn f(&mut self, value: f32) -> Id {
        self.constant(self.float, value.to_bits())
    }

    // ---- Global variables ----

    /// A storage buffer at a binding of set 0
    pub fn storage_buffer(&mut self, binding: u32) -> Buffer {
        let array = self.type_of(OP_TYPE_RUNTIME_ARRAY, &[self.uint]);
        if !self.decorations.windows(4).any(|words| words == [(4 << 16) | OP_DECORATE, array, DECORATE_ARRAY_STRIDE, 4]) {
            instruction(&mut self.decorations, OP_DECORATE, &[array, DECORATE_ARRAY_STRIDE, 4]);
        }
        // Each its own struct type: decorations belong to it
        let block = self.id();
        instruction(&mut self.globals, OP_TYPE_STRUCT, &[block, array]);
        instruction(&mut self.decorations, OP_DECORATE, &[block, DECORATE_BUFFER_BLOCK]);
        instruction(&mut self.decorations, OP_MEMBER_DECORATE, &[block, 0, DECORATE_OFFSET, 0]);
        let pointer = self.pointer_to(STORAGE_UNIFORM, block);
        let variable = self.id();
        instruction(&mut self.globals, OP_VARIABLE, &[pointer, variable, STORAGE_UNIFORM]);
        instruction(&mut self.decorations, OP_DECORATE, &[variable, DECORATE_DESCRIPTOR_SET, 0]);
        instruction(&mut self.decorations, OP_DECORATE, &[variable, DECORATE_BINDING, binding]);
        Buffer(variable)
    }

    pub fn push_constants(&mut self, members: &[Id]) -> Push {
        let block = self.id();
        let mut words = vec![block];
        words.extend_from_slice(members);
        instruction(&mut self.globals, OP_TYPE_STRUCT, &words);
        instruction(&mut self.decorations, OP_DECORATE, &[block, DECORATE_BLOCK]);
        for index in 0..members.len() as u32 {
            instruction(&mut self.decorations, OP_MEMBER_DECORATE, &[block, index, DECORATE_OFFSET, index * 4]);
        }
        let pointer = self.pointer_to(STORAGE_PUSH_CONSTANT, block);
        let id = self.id();
        instruction(&mut self.globals, OP_VARIABLE, &[pointer, id, STORAGE_PUSH_CONSTANT]);
        Push { id, members: members.to_vec() }
    }

    /// A variable each invocation has its own of
    pub fn private(&mut self, of: Id) -> Variable {
        let pointer = self.pointer_to(STORAGE_PRIVATE, of);
        let id = self.id();
        instruction(&mut self.globals, OP_VARIABLE, &[pointer, id, STORAGE_PRIVATE]);
        Variable { id, of }
    }

    /// An array each invocation has its own of, starting as `values` (the bits of each), or zeros
    /// if `values` has only its length to say
    pub fn table(&mut self, of: Id, length: u32, values: Option<&[u32]>) -> Table {
        let size = self.u(length);
        let array = self.type_of(OP_TYPE_ARRAY, &[of, size]);
        let pointer = self.pointer_to(STORAGE_PRIVATE, array);
        let id = match values {
            Some(values) => {
                assert_eq!(values.len() as u32, length);
                let parts: Vec<Id> = values.iter().map(|&bits| self.constant(of, bits)).collect();
                let initial = self.id();
                let mut words = vec![array, initial];
                words.extend_from_slice(&parts);
                instruction(&mut self.globals, OP_CONSTANT_COMPOSITE, &words);
                let id = self.id();
                instruction(&mut self.globals, OP_VARIABLE, &[pointer, id, STORAGE_PRIVATE, initial]);
                id
            }
            None => {
                let id = self.id();
                instruction(&mut self.globals, OP_VARIABLE, &[pointer, id, STORAGE_PRIVATE]);
                id
            }
        };
        Table { id, of }
    }

    /// Reads gl_GlobalInvocationID's x (0), y (1) or z (2)
    pub fn invocation(&mut self, component: u32) -> Id {
        let vector = self.type_of(OP_TYPE_VECTOR, &[self.uint, 3]);
        if self.invocation_id == 0 {
            let pointer = self.pointer_to(STORAGE_INPUT, vector);
            let id = self.id();
            instruction(&mut self.globals, OP_VARIABLE, &[pointer, id, STORAGE_INPUT]);
            instruction(&mut self.decorations, OP_DECORATE, &[id, DECORATE_BUILT_IN, BUILT_IN_GLOBAL_INVOCATION_ID]);
            self.invocation_id = id;
        }
        let whole = self.value(vector, OP_LOAD, &[self.invocation_id]);
        self.value(self.uint, OP_COMPOSITE_EXTRACT, &[whole, component])
    }

    // ---- Values ----

    fn value(&mut self, of: Id, op: u32, operands: &[u32]) -> Id {
        let id = self.id();
        let mut words = vec![of, id];
        words.extend_from_slice(operands);
        instruction(&mut self.body, op, &words);
        id
    }

    fn extended(&mut self, of: Id, op: u32, operands: &[u32]) -> Id {
        let mut words = vec![self.glsl, op];
        words.extend_from_slice(operands);
        self.value(of, OP_EXT_INST, &words)
    }

    pub fn local(&mut self, of: Id) -> Variable {
        let pointer = self.pointer_to(STORAGE_FUNCTION, of);
        let id = self.id();
        instruction(&mut self.locals, OP_VARIABLE, &[pointer, id, STORAGE_FUNCTION]);
        Variable { id, of }
    }

    /// A local starting at `value`
    pub fn local_of(&mut self, of: Id, value: Id) -> Variable {
        let variable = self.local(of);
        self.store(variable, value);
        variable
    }

    pub fn load(&mut self, variable: Variable) -> Id {
        self.value(variable.of, OP_LOAD, &[variable.id])
    }

    pub fn store(&mut self, variable: Variable, value: Id) {
        instruction(&mut self.body, OP_STORE, &[variable.id, value]);
    }

    pub fn read_push(&mut self, push: &Push, index: u32) -> Id {
        let of = push.members[index as usize];
        let pointer = self.pointer_to(STORAGE_PUSH_CONSTANT, of);
        let at = self.u(index);
        let element = self.value(pointer, OP_ACCESS_CHAIN, &[push.id, at]);
        self.value(of, OP_LOAD, &[element])
    }

    fn element(&mut self, buffer: Buffer, at: Id) -> Id {
        let pointer = self.pointer_to(STORAGE_UNIFORM, self.uint);
        let member = self.u(0);
        self.value(pointer, OP_ACCESS_CHAIN, &[buffer.0, member, at])
    }

    pub fn read(&mut self, buffer: Buffer, at: Id) -> Id {
        let element = self.element(buffer, at);
        self.value(self.uint, OP_LOAD, &[element])
    }

    pub fn write(&mut self, buffer: Buffer, at: Id, value: Id) {
        let element = self.element(buffer, at);
        instruction(&mut self.body, OP_STORE, &[element, value]);
    }

    fn table_element(&mut self, table: Table, at: Id) -> Id {
        let pointer = self.pointer_to(STORAGE_PRIVATE, table.of);
        self.value(pointer, OP_ACCESS_CHAIN, &[table.id, at])
    }

    pub fn get(&mut self, table: Table, at: Id) -> Id {
        let element = self.table_element(table, at);
        self.value(table.of, OP_LOAD, &[element])
    }

    pub fn set(&mut self, table: Table, at: Id, value: Id) {
        let element = self.table_element(table, at);
        instruction(&mut self.body, OP_STORE, &[element, value]);
    }

    pub fn add(&mut self, a: Id, b: Id) -> Id {
        self.value(self.uint, OP_I_ADD, &[a, b])
    }

    pub fn sub(&mut self, a: Id, b: Id) -> Id {
        self.value(self.uint, OP_I_SUB, &[a, b])
    }

    pub fn mul(&mut self, a: Id, b: Id) -> Id {
        self.value(self.uint, OP_I_MUL, &[a, b])
    }

    pub fn div(&mut self, a: Id, b: Id) -> Id {
        self.value(self.uint, OP_U_DIV, &[a, b])
    }

    pub fn shl(&mut self, a: Id, bits: Id) -> Id {
        self.value(self.uint, OP_SHIFT_LEFT_LOGICAL, &[a, bits])
    }

    pub fn shr(&mut self, a: Id, bits: Id) -> Id {
        self.value(self.uint, OP_SHIFT_RIGHT_LOGICAL, &[a, bits])
    }

    pub fn and(&mut self, a: Id, b: Id) -> Id {
        self.value(self.uint, OP_BITWISE_AND, &[a, b])
    }

    pub fn or(&mut self, a: Id, b: Id) -> Id {
        self.value(self.uint, OP_BITWISE_OR, &[a, b])
    }

    pub fn xor(&mut self, a: Id, b: Id) -> Id {
        self.value(self.uint, OP_BITWISE_XOR, &[a, b])
    }

    pub fn umin(&mut self, a: Id, b: Id) -> Id {
        self.extended(self.uint, GLSL_U_MIN, &[a, b])
    }

    /// The position of the highest set bit
    pub fn msb(&mut self, a: Id) -> Id {
        self.extended(self.uint, GLSL_FIND_U_MSB, &[a])
    }

    /// The position of the lowest set bit
    pub fn lsb(&mut self, a: Id) -> Id {
        self.extended(self.uint, GLSL_FIND_I_LSB, &[a])
    }

    /// `a` if `condition` holds, otherwise `b`, with no branch
    pub fn select(&mut self, of: Id, condition: Id, a: Id, b: Id) -> Id {
        self.value(of, OP_SELECT, &[condition, a, b])
    }

    pub fn lt(&mut self, a: Id, b: Id) -> Id {
        self.value(self.boolean, OP_U_LESS_THAN, &[a, b])
    }

    pub fn eq(&mut self, a: Id, b: Id) -> Id {
        self.value(self.boolean, OP_I_EQUAL, &[a, b])
    }

    pub fn ne(&mut self, a: Id, b: Id) -> Id {
        self.value(self.boolean, OP_I_NOT_EQUAL, &[a, b])
    }

    pub fn not(&mut self, a: Id) -> Id {
        self.value(self.boolean, OP_LOGICAL_NOT, &[a])
    }

    pub fn both(&mut self, a: Id, b: Id) -> Id {
        self.value(self.boolean, OP_LOGICAL_AND, &[a, b])
    }

    // Signed

    pub fn isub(&mut self, a: Id, b: Id) -> Id {
        self.value(self.int, OP_I_SUB, &[a, b])
    }

    pub fn iclamp(&mut self, a: Id, low: Id, high: Id) -> Id {
        self.extended(self.int, GLSL_S_CLAMP, &[a, low, high])
    }

    /// The bits of a signed value, as unsigned
    pub fn as_uint(&mut self, a: Id) -> Id {
        self.value(self.uint, OP_BITCAST, &[a])
    }

    /// The top 16 bits of a word, as a signed number
    pub fn arithmetic_shift_16(&mut self, word: Id) -> Id {
        let signed = self.value(self.int, OP_BITCAST, &[word]);
        let bits = self.i(16);
        self.value(self.int, OP_SHIFT_RIGHT_ARITHMETIC, &[signed, bits])
    }

    /// A signed value's sign, all ones or all zeros, as unsigned
    pub fn sign_mask(&mut self, a: Id) -> Id {
        let bits = self.i(31);
        let mask = self.value(self.int, OP_SHIFT_RIGHT_ARITHMETIC, &[a, bits]);
        self.as_uint(mask)
    }

    // Floats

    pub fn fadd(&mut self, a: Id, b: Id) -> Id {
        self.value(self.float, OP_F_ADD, &[a, b])
    }

    pub fn fsub(&mut self, a: Id, b: Id) -> Id {
        self.value(self.float, OP_F_SUB, &[a, b])
    }

    pub fn fmul(&mut self, a: Id, b: Id) -> Id {
        self.value(self.float, OP_F_MUL, &[a, b])
    }

    pub fn fdiv(&mut self, a: Id, b: Id) -> Id {
        self.value(self.float, OP_F_DIV, &[a, b])
    }

    pub fn to_float(&mut self, a: Id) -> Id {
        self.value(self.float, OP_CONVERT_U_TO_F, &[a])
    }

    /// Rounded to the nearest whole number
    pub fn round(&mut self, a: Id) -> Id {
        let rounded = self.extended(self.float, GLSL_ROUND, &[a]);
        self.value(self.int, OP_CONVERT_F_TO_S, &[rounded])
    }

    // ---- Control flow ----

    fn label(&mut self, id: Id) {
        instruction(&mut self.body, OP_LABEL, &[id]);
    }

    fn branch(&mut self, to: Id) {
        instruction(&mut self.body, OP_BRANCH, &[to]);
    }

    pub fn if_(&mut self, condition: Id, then: impl FnOnce(&mut Self)) {
        let (then_label, merge) = (self.id(), self.id());
        instruction(&mut self.body, OP_SELECTION_MERGE, &[merge, 0]);
        instruction(&mut self.body, OP_BRANCH_CONDITIONAL, &[condition, then_label, merge]);
        self.label(then_label);
        then(self);
        self.branch(merge);
        self.label(merge);
    }

    pub fn if_else(&mut self, condition: Id, then: impl FnOnce(&mut Self), otherwise: impl FnOnce(&mut Self)) {
        let (then_label, else_label, merge) = (self.id(), self.id(), self.id());
        instruction(&mut self.body, OP_SELECTION_MERGE, &[merge, 0]);
        instruction(&mut self.body, OP_BRANCH_CONDITIONAL, &[condition, then_label, else_label]);
        self.label(then_label);
        then(self);
        self.branch(merge);
        self.label(else_label);
        otherwise(self);
        self.branch(merge);
        self.label(merge);
    }

    pub fn while_(&mut self, condition: impl FnOnce(&mut Self) -> Id, body: impl FnOnce(&mut Self)) {
        let (header, check, body_label, next, merge) = (self.id(), self.id(), self.id(), self.id(), self.id());
        self.branch(header);
        self.label(header);
        instruction(&mut self.body, OP_LOOP_MERGE, &[merge, next, 0]);
        self.branch(check);
        self.label(check);
        let holds = condition(self);
        instruction(&mut self.body, OP_BRANCH_CONDITIONAL, &[holds, body_label, merge]);
        self.label(body_label);
        body(self);
        self.branch(next);
        self.label(next);
        self.branch(header);
        self.label(merge);
    }

    /// `body` for each of from, from + step, ... below `to`
    pub fn for_(&mut self, from: u32, to: Id, step: u32, body: impl FnOnce(&mut Self, Id)) {
        let start = self.u(from);
        let counter = self.local_of(self.uint, start);
        self.while_(
            |s| {
                let at = s.load(counter);
                s.lt(at, to)
            },
            |s| {
                let at = s.load(counter);
                body(s, at);
                let again = s.load(counter);
                let step = s.u(step);
                let next = s.add(again, step);
                s.store(counter, next);
            },
        );
    }

    // ---- Functions ----

    /// Defines a function; `body` gets its parameters and returns what it returns. Functions can't be
    /// defined inside one another, and a function must be defined before it's called.
    pub fn function(&mut self, returns: Id, parameters: &[Id], body: impl FnOnce(&mut Self, &[Id]) -> Option<Id>) -> Id {
        assert!(self.body.is_empty() && self.locals.is_empty());
        let mut signature = vec![returns];
        signature.extend_from_slice(parameters);
        let kind = self.type_of(OP_TYPE_FUNCTION, &signature);
        let function = self.id();
        instruction(&mut self.functions, OP_FUNCTION, &[returns, function, 0, kind]);
        let mut values = Vec::with_capacity(parameters.len());
        for &of in parameters {
            let id = self.id();
            instruction(&mut self.functions, OP_FUNCTION_PARAMETER, &[of, id]);
            values.push(id);
        }
        let first = self.id();
        instruction(&mut self.functions, OP_LABEL, &[first]);
        let returned = body(self, &values);
        match returned {
            Some(value) => instruction(&mut self.body, OP_RETURN_VALUE, &[value]),
            None => instruction(&mut self.body, OP_RETURN, &[]),
        }
        instruction(&mut self.body, OP_FUNCTION_END, &[]);
        let (locals, code) = (std::mem::take(&mut self.locals), std::mem::take(&mut self.body));
        self.functions.extend_from_slice(&locals);
        self.functions.extend_from_slice(&code);
        function
    }

    pub fn call(&mut self, returns: Id, function: Id, arguments: &[Id]) -> Id {
        let mut words = vec![function];
        words.extend_from_slice(arguments);
        self.value(returns, OP_FUNCTION_CALL, &words)
    }

    /// Defines the entry point, of one invocation per workgroup
    pub fn main(&mut self, body: impl FnOnce(&mut Self)) {
        self.entry = self.function(self.void, &[], |s, _| {
            body(s);
            None
        });
    }

    /// The finished module
    pub fn finish(self) -> Vec<u32> {
        const NAME_MAIN: u32 = 0x6E69_616D;
        const GLSL_NAME: [u32; 4] = [0x4C53_4C47, 0x6474_732E, 0x3035_342E, 0];
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, self.next, 0];
        instruction(&mut words, OP_CAPABILITY, &[1]);
        let mut import = vec![self.glsl];
        import.extend_from_slice(&GLSL_NAME);
        instruction(&mut words, OP_EXT_INST_IMPORT, &import);
        instruction(&mut words, OP_MEMORY_MODEL, &[0, 1]);
        let mut entry = vec![5, self.entry, NAME_MAIN, 0];
        if self.invocation_id != 0 {
            entry.push(self.invocation_id);
        }
        instruction(&mut words, OP_ENTRY_POINT, &entry);
        instruction(&mut words, OP_EXECUTION_MODE, &[self.entry, 17, 1, 1, 1]);
        words.extend_from_slice(&self.decorations);
        words.extend_from_slice(&self.globals);
        words.extend_from_slice(&self.functions);
        words
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every instruction's length must land on the next one
    fn instructions(words: &[u32]) -> Vec<u32> {
        let mut at = 5;
        let mut ops = Vec::new();
        while at < words.len() {
            let length = (words[at] >> 16) as usize;
            assert!(length > 0 && at + length <= words.len(), "at word {at}");
            ops.push(words[at] & 0xFFFF);
            at += length;
        }
        ops
    }

    #[test]
    fn a_module_has_its_sections_in_order() {
        let mut s = Shader::new();
        let buffer = s.storage_buffer(0);
        s.main(|s| {
            let x = s.invocation(0);
            let total = s.local_of(s.uint, x);
            let ten = s.u(10);
            s.for_(0, ten, 2, |s, i| {
                let now = s.load(total);
                let more = s.add(now, i);
                s.store(total, more);
            });
            let value = s.load(total);
            let zero = s.u(0);
            s.write(buffer, zero, value);
        });
        let words = s.finish();
        assert_eq!(words[0], 0x0723_0203);
        let ops = instructions(&words);
        let position = |op: u32| ops.iter().position(|&o| o == op).unwrap();
        assert!(position(OP_CAPABILITY) < position(OP_EXT_INST_IMPORT));
        assert!(position(OP_EXT_INST_IMPORT) < position(OP_MEMORY_MODEL));
        assert!(position(OP_MEMORY_MODEL) < position(OP_ENTRY_POINT));
        assert!(position(OP_ENTRY_POINT) < position(OP_EXECUTION_MODE));
        assert!(position(OP_EXECUTION_MODE) < position(OP_DECORATE));
        assert!(position(OP_DECORATE) < position(OP_TYPE_VOID));
        assert!(position(OP_TYPE_VOID) < position(OP_FUNCTION));
        // Each result id is below the module's bound
        assert!(words[3] > 1);
    }
}
