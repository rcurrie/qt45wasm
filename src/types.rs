use serde::{Deserialize, Serialize};

/// A WASM value type
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValType {
    I32,
    I64,
    F32,
    F64,
}

impl std::fmt::Display for ValType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValType::I32 => write!(f, "i32"),
            ValType::I64 => write!(f, "i64"),
            ValType::F32 => write!(f, "f32"),
            ValType::F64 => write!(f, "f64"),
        }
    }
}

/// Function signature: parameter types and return types
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSignature {
    pub params: Vec<ValType>,
    pub results: Vec<ValType>,
}

impl std::fmt::Display for FunctionSignature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        for (i, p) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", p)?;
        }
        write!(f, ") -> (")?;
        for (i, r) in self.results.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", r)?;
        }
        write!(f, ")")
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
