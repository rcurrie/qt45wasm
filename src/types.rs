use wasmtime::Val;

/// A runtime value that can be passed to/from WASM functions.
#[derive(Debug, Clone)]
pub enum Value {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

impl Value {
    /// Convert to a wasmtime Val for dynamic function calls.
    pub fn to_val(&self) -> Val {
        match self {
            Value::I32(v) => Val::I32(*v),
            Value::I64(v) => Val::I64(*v),
            Value::F32(v) => Val::F32(v.to_bits()),
            Value::F64(v) => Val::F64(v.to_bits()),
        }
    }

    /// Convert from a wasmtime Val.
    pub fn from_val(val: &Val) -> anyhow::Result<Self> {
        match val {
            Val::I32(v) => Ok(Value::I32(*v)),
            Val::I64(v) => Ok(Value::I64(*v)),
            Val::F32(v) => Ok(Value::F32(f32::from_bits(*v))),
            Val::F64(v) => Ok(Value::F64(f64::from_bits(*v))),
            other => anyhow::bail!("Unsupported WASM value type: {:?}", other),
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::I32(v) => write!(f, "{v}"),
            Value::I64(v) => write!(f, "{v}"),
            Value::F32(v) => write!(f, "{v}"),
            Value::F64(v) => write!(f, "{v}"),
        }
    }
}

/// A stored function record from the database
#[allow(dead_code)]
#[derive(Debug)]
pub struct StoredFunction {
    pub id: i64,
    pub name: String,
    pub description: String,
    pub signature: String,
    pub source_lang: String,
    pub source_code: Option<String>,
    pub wasm_binary: Option<Vec<u8>>,
    pub call_count: i64,
    pub is_verified: bool,
}
