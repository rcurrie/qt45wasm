use anyhow::{Context, Result};
use wasmtime::{Engine, Instance, Module, Store, Val};

use crate::types::Value;

/// WASM compilation and execution engine.
pub struct WasmRuntime {
    engine: Engine,
}

impl WasmRuntime {
    pub fn new() -> Result<Self> {
        let engine = Engine::default();
        Ok(Self { engine })
    }

    /// Compile WAT source text into a WASM module.
    /// Returns the module and its serialized bytes for caching.
    pub fn compile_wat(&self, wat: &str) -> Result<(Module, Vec<u8>)> {
        let module = Module::new(&self.engine, wat).context("Failed to compile WAT")?;
        let bytes = module.serialize().context("Failed to serialize module")?;
        Ok((module, bytes))
    }

    /// Load a module from pre-compiled bytes (stored in the DB).
    /// This is extremely fast — skips all parsing and compilation.
    pub fn load_cached(&self, bytes: &[u8]) -> Result<Module> {
        // SAFETY: bytes come from our own Module::serialize() stored in SQLite.
        // We control the entire pipeline from serialize to deserialize.
        let module =
            unsafe { Module::deserialize(&self.engine, bytes) }.context("Failed to deserialize cached module")?;
        Ok(module)
    }

    /// Execute a function by name with dynamic argument and return types.
    pub fn call(&self, module: &Module, func_name: &str, args: &[Value]) -> Result<Vec<Value>> {
        let mut store = Store::new(&self.engine, ());
        let instance = Instance::new(&mut store, module, &[])?;

        let func = instance
            .get_func(&mut store, func_name)
            .ok_or_else(|| anyhow::anyhow!("Function '{func_name}' not exported by module"))?;

        let params: Vec<Val> = args.iter().map(|v| v.to_val()).collect();
        let result_count = func.ty(&store).results().len();
        let mut results = vec![Val::I32(0); result_count];

        func.call(&mut store, &params, &mut results)?;

        results
            .iter()
            .map(Value::from_val)
            .collect()
    }
}
