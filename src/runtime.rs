use anyhow::{Context, Result};
use wasmtime::{Engine, Instance, Module, Store};

/// WASM compilation and execution engine.
pub struct WasmRuntime {
    engine: Engine,
}

impl WasmRuntime {
    pub fn new() -> Result<Self> {
        let engine = Engine::default();
        Ok(Self { engine })
    }

    #[allow(dead_code)]
    pub fn engine(&self) -> &Engine {
        &self.engine
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

    /// Execute a function by name from a module with i32 arguments.
    /// Returns the i32 result.
    pub fn call_i32(&self, module: &Module, func_name: &str, args: &[i32]) -> Result<i32> {
        let mut store = Store::new(&self.engine, ());
        let instance = Instance::new(&mut store, module, &[])?;

        match args.len() {
            0 => {
                let func = instance.get_typed_func::<(), i32>(&mut store, func_name)?;
                Ok(func.call(&mut store, ())?)
            }
            1 => {
                let func = instance.get_typed_func::<i32, i32>(&mut store, func_name)?;
                Ok(func.call(&mut store, args[0])?)
            }
            2 => {
                let func = instance.get_typed_func::<(i32, i32), i32>(&mut store, func_name)?;
                Ok(func.call(&mut store, (args[0], args[1]))?)
            }
            3 => {
                let func = instance.get_typed_func::<(i32, i32, i32), i32>(&mut store, func_name)?;
                Ok(func.call(&mut store, (args[0], args[1], args[2]))?)
            }
            n => anyhow::bail!("Unsupported argument count: {n} (max 3 for i32 calls)"),
        }
    }
}
