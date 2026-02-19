use anyhow::{bail, Result};
use serde_json::json;
use wasmtime::Val;

/// Compute tier for a stored function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComputeTier {
    Scalar,
    Simd,
    Gpu,
}

impl ComputeTier {
    pub fn as_str(&self) -> &'static str {
        match self {
            ComputeTier::Scalar => "scalar",
            ComputeTier::Simd => "simd",
            ComputeTier::Gpu => "gpu",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "simd" => ComputeTier::Simd,
            "gpu" => ComputeTier::Gpu,
            _ => ComputeTier::Scalar,
        }
    }
}

impl std::fmt::Display for ComputeTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Array data for SIMD and GPU functions.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum ArrayValue {
    F32Array(Vec<f32>),
    F64Array(Vec<f64>),
    I32Array(Vec<i32>),
}

impl ArrayValue {
    /// Number of elements in the array.
    pub fn element_count(&self) -> usize {
        match self {
            ArrayValue::F32Array(v) => v.len(),
            ArrayValue::F64Array(v) => v.len(),
            ArrayValue::I32Array(v) => v.len(),
        }
    }

    /// Byte size of elements (before padding).
    fn raw_byte_len(&self) -> usize {
        match self {
            ArrayValue::F32Array(v) => v.len() * 4,
            ArrayValue::F64Array(v) => v.len() * 8,
            ArrayValue::I32Array(v) => v.len() * 4,
        }
    }

    /// Byte length, padded up to 16-byte boundary for v128 alignment.
    pub fn byte_len(&self) -> usize {
        let raw = self.raw_byte_len();
        (raw + 15) & !15
    }

    /// Serialize to little-endian bytes, padded to 16-byte boundary.
    pub fn to_bytes(&self) -> Vec<u8> {
        let padded = self.byte_len();
        let mut bytes = Vec::with_capacity(padded);
        match self {
            ArrayValue::F32Array(v) => {
                for x in v {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
            }
            ArrayValue::F64Array(v) => {
                for x in v {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
            }
            ArrayValue::I32Array(v) => {
                for x in v {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
        bytes.resize(padded, 0);
        bytes
    }

    /// Deserialize f32 array from little-endian bytes.
    pub fn from_bytes_f32(bytes: &[u8], count: usize) -> Self {
        let mut v = Vec::with_capacity(count);
        for i in 0..count {
            let start = i * 4;
            let b = [bytes[start], bytes[start + 1], bytes[start + 2], bytes[start + 3]];
            v.push(f32::from_le_bytes(b));
        }
        ArrayValue::F32Array(v)
    }

    /// Serialize to JSON for test case storage.
    #[cfg(test)]
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            ArrayValue::F32Array(v) => json!({"type": "f32[]", "value": v}),
            ArrayValue::F64Array(v) => json!({"type": "f64[]", "value": v}),
            ArrayValue::I32Array(v) => json!({"type": "i32[]", "value": v}),
        }
    }

    /// Deserialize from JSON.
    #[cfg(test)]
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let ty = v["type"].as_str().unwrap_or("");
        let arr = v["value"].as_array().ok_or_else(|| anyhow::anyhow!("Expected array value"))?;
        match ty {
            "f32[]" => {
                let vals: Vec<f32> = arr.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
                Ok(ArrayValue::F32Array(vals))
            }
            "f64[]" => {
                let vals: Vec<f64> = arr.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect();
                Ok(ArrayValue::F64Array(vals))
            }
            "i32[]" => {
                let vals: Vec<i32> = arr.iter().map(|x| x.as_i64().unwrap_or(0) as i32).collect();
                Ok(ArrayValue::I32Array(vals))
            }
            _ => bail!("Unknown array type in JSON: '{ty}'"),
        }
    }

    /// Check approximate equality for f32 arrays.
    pub fn approx_eq(&self, other: &ArrayValue) -> bool {
        match (self, other) {
            (ArrayValue::F32Array(a), ArrayValue::F32Array(b)) => {
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 0.001)
            }
            (ArrayValue::F64Array(a), ArrayValue::F64Array(b)) => {
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 0.001)
            }
            (ArrayValue::I32Array(a), ArrayValue::I32Array(b)) => a == b,
            _ => false,
        }
    }
}

impl std::fmt::Display for ArrayValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArrayValue::F32Array(v) => {
                let strs: Vec<String> = v.iter().map(|x| format!("{x}")).collect();
                write!(f, "[{}]", strs.join(", "))
            }
            ArrayValue::F64Array(v) => {
                let strs: Vec<String> = v.iter().map(|x| format!("{x}")).collect();
                write!(f, "[{}]", strs.join(", "))
            }
            ArrayValue::I32Array(v) => {
                let strs: Vec<String> = v.iter().map(|x| format!("{x}")).collect();
                write!(f, "[{}]", strs.join(", "))
            }
        }
    }
}

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

/// A test case for array-based (SIMD/GPU) functions.
#[derive(Debug)]
pub struct ArrayTestCase {
    #[allow(dead_code)]
    pub id: Option<i64>,
    pub inputs: Vec<ArrayValue>,
    pub expected: ArrayValue,
}

/// A stored function record from the database
#[allow(dead_code)]
#[derive(Debug, Clone)]
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
    pub compute_tier: ComputeTier,
    pub shader_source: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_value_to_bytes_padding() {
        // 3 floats = 12 bytes, should pad to 16
        let arr = ArrayValue::F32Array(vec![1.0, 2.0, 3.0]);
        let bytes = arr.to_bytes();
        assert_eq!(bytes.len(), 16);
        // Last 4 bytes should be zeros (padding)
        assert_eq!(&bytes[12..16], &[0, 0, 0, 0]);
    }

    #[test]
    fn test_array_value_to_bytes_aligned() {
        // 4 floats = 16 bytes, already aligned
        let arr = ArrayValue::F32Array(vec![1.0, 2.0, 3.0, 4.0]);
        let bytes = arr.to_bytes();
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn test_array_value_roundtrip_f32() {
        let original = vec![1.0f32, 2.5, -3.0, 4.0];
        let arr = ArrayValue::F32Array(original.clone());
        let bytes = arr.to_bytes();
        let restored = ArrayValue::from_bytes_f32(&bytes, 4);
        match restored {
            ArrayValue::F32Array(v) => assert_eq!(v, original),
            _ => panic!("Expected F32Array"),
        }
    }

    #[test]
    fn test_array_value_json_roundtrip() {
        let arr = ArrayValue::F32Array(vec![1.0, 2.0, 3.0]);
        let json = arr.to_json();
        let restored = ArrayValue::from_json(&json).unwrap();
        assert!(arr.approx_eq(&restored));
    }

    #[test]
    fn test_compute_tier_roundtrip() {
        assert_eq!(ComputeTier::from_str(ComputeTier::Scalar.as_str()), ComputeTier::Scalar);
        assert_eq!(ComputeTier::from_str(ComputeTier::Simd.as_str()), ComputeTier::Simd);
        assert_eq!(ComputeTier::from_str(ComputeTier::Gpu.as_str()), ComputeTier::Gpu);
    }
}
