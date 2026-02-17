use anyhow::{bail, Result};
use serde_json::json;
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
    pub fn from_val(val: &Val) -> Result<Self> {
        match val {
            Val::I32(v) => Ok(Value::I32(*v)),
            Val::I64(v) => Ok(Value::I64(*v)),
            Val::F32(v) => Ok(Value::F32(f32::from_bits(*v))),
            Val::F64(v) => Ok(Value::F64(f64::from_bits(*v))),
            other => bail!("Unsupported WASM value type: {:?}", other),
        }
    }

    /// Serialize to JSON for test case storage.
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Value::I32(v) => json!({"type": "i32", "value": v}),
            Value::I64(v) => json!({"type": "i64", "value": v}),
            Value::F32(v) => json!({"type": "f32", "value": v}),
            Value::F64(v) => json!({"type": "f64", "value": v}),
        }
    }

    /// Deserialize from JSON test case format.
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let ty = v["type"].as_str().unwrap_or("");
        match ty {
            "i32" => Ok(Value::I32(v["value"].as_i64().unwrap_or(0) as i32)),
            "i64" => Ok(Value::I64(v["value"].as_i64().unwrap_or(0))),
            "f32" => Ok(Value::F32(v["value"].as_f64().unwrap_or(0.0) as f32)),
            "f64" => Ok(Value::F64(v["value"].as_f64().unwrap_or(0.0))),
            _ => bail!("Unknown value type in JSON: '{ty}'"),
        }
    }

    /// Check approximate equality (exact for integers, epsilon for floats).
    pub fn approx_eq(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::I32(a), Value::I32(b)) => a == b,
            (Value::I64(a), Value::I64(b)) => a == b,
            (Value::F32(a), Value::F32(b)) => (a - b).abs() < 0.001,
            (Value::F64(a), Value::F64(b)) => (a - b).abs() < 0.001,
            _ => false,
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

/// A test case for verifying a function's correctness.
#[derive(Debug)]
pub struct TestCase {
    #[allow(dead_code)]
    pub id: Option<i64>,
    pub input: Vec<Value>,
    pub expected: Vec<Value>,
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
