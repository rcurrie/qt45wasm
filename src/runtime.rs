use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use wasmtime::{Caller, Engine, Instance, Linker, Module, Store, Val};

use crate::gpu::GpuContext;
use crate::types::{ArrayValue, Value};

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

    /// Execute a scalar function by name with dynamic argument and return types.
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

    /// Execute an array-consuming function (SIMD tier).
    ///
    /// Writes input arrays into the module's exported linear memory sequentially,
    /// calls the function with (data_ptr, element_count), and reads the result
    /// array back from the returned offset.
    pub fn call_array(
        &self,
        module: &Module,
        func_name: &str,
        inputs: &[ArrayValue],
        result_count: usize,
    ) -> Result<ArrayValue> {
        let mut store = Store::new(&self.engine, ());
        let linker = Linker::<()>::new(&self.engine);
        let instance = linker.instantiate(&mut store, module)?;

        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| anyhow::anyhow!("Module has no exported 'memory'"))?;

        // Write input arrays sequentially into memory at offset 0
        let mut offset: usize = 0;
        let mut element_count: usize = 0;
        for input in inputs {
            let bytes = input.to_bytes();
            let needed = offset + bytes.len();

            // Grow memory if needed
            let current_size = memory.data_size(&store);
            if needed > current_size {
                let pages_needed = ((needed - current_size) + 65535) / 65536;
                memory.grow(&mut store, pages_needed as u64)?;
            }

            let mem_data = memory.data_mut(&mut store);
            mem_data[offset..offset + bytes.len()].copy_from_slice(&bytes);
            element_count = input.element_count();
            offset += bytes.len();
        }

        // Call the function with (data_ptr=0, element_count)
        let func = instance
            .get_func(&mut store, func_name)
            .ok_or_else(|| anyhow::anyhow!("Function '{func_name}' not exported"))?;
        let params = vec![Val::I32(0), Val::I32(element_count as i32)];
        let mut results = vec![Val::I32(0)];
        func.call(&mut store, &params, &mut results)?;

        let result_ptr = match results[0] {
            Val::I32(v) => v as usize,
            _ => anyhow::bail!("Expected i32 result pointer from array function"),
        };

        // Read result back from memory
        let result_byte_len = result_count * 4;
        let mem_data = memory.data(&store);
        if result_ptr + result_byte_len > mem_data.len() {
            anyhow::bail!(
                "Result at offset {} with size {} exceeds memory bounds {}",
                result_ptr, result_byte_len, mem_data.len()
            );
        }
        let result_bytes = &mem_data[result_ptr..result_ptr + result_byte_len];
        Ok(ArrayValue::from_bytes_f32(result_bytes, result_count))
    }

    /// Execute a GPU-tier function.
    ///
    /// Registers host functions (gpu.alloc, gpu.write_buffer, gpu.dispatch_shader,
    /// gpu.read_buffer) via the wasmtime Linker, then writes input data into
    /// WASM memory, calls the function, and reads results back.
    pub fn call_gpu(
        &self,
        module: &Module,
        func_name: &str,
        gpu: &Arc<Mutex<GpuContext>>,
        inputs: &[ArrayValue],
        result_count: usize,
    ) -> Result<ArrayValue> {
        let mut store = Store::new(&self.engine, ());
        let mut linker = Linker::<()>::new(&self.engine);

        // gpu.alloc(size: i32) -> handle: i32
        let gpu_alloc = Arc::clone(gpu);
        linker.func_wrap("gpu", "alloc", move |size: i32| -> i32 {
            gpu_alloc.lock().unwrap().alloc(size as usize)
        })?;

        // gpu.write_buffer(handle: i32, ptr: i32, len: i32)
        let gpu_write = Arc::clone(gpu);
        linker.func_wrap(
            "gpu",
            "write_buffer",
            move |mut caller: Caller<'_, ()>, handle: i32, ptr: i32, len: i32| {
                let memory = caller.get_export("memory").unwrap().into_memory().unwrap();
                let data = &memory.data(&caller)[ptr as usize..(ptr as usize + len as usize)];
                let _ = gpu_write.lock().unwrap().write_buffer(handle, data);
            },
        )?;

        // gpu.dispatch_shader(shader_ptr: i32, shader_len: i32, buf_handle: i32, workgroups: i32)
        let gpu_dispatch = Arc::clone(gpu);
        linker.func_wrap(
            "gpu",
            "dispatch_shader",
            move |mut caller: Caller<'_, ()>,
                  shader_ptr: i32,
                  shader_len: i32,
                  buf_handle: i32,
                  workgroups: i32| {
                let memory = caller.get_export("memory").unwrap().into_memory().unwrap();
                let wgsl_bytes = &memory.data(&caller)
                    [shader_ptr as usize..(shader_ptr as usize + shader_len as usize)];
                let wgsl = std::str::from_utf8(wgsl_bytes).unwrap();
                let _ = gpu_dispatch
                    .lock()
                    .unwrap()
                    .dispatch_shader(wgsl, buf_handle, workgroups as u32);
            },
        )?;

        // gpu.read_buffer(handle: i32, ptr: i32, len: i32)
        let gpu_read = Arc::clone(gpu);
        linker.func_wrap(
            "gpu",
            "read_buffer",
            move |mut caller: Caller<'_, ()>, handle: i32, ptr: i32, len: i32| {
                let result = gpu_read
                    .lock()
                    .unwrap()
                    .read_buffer(handle, len as usize)
                    .unwrap();
                let memory = caller.get_export("memory").unwrap().into_memory().unwrap();
                memory.data_mut(&mut caller)[ptr as usize..(ptr as usize + len as usize)]
                    .copy_from_slice(&result);
            },
        )?;

        let instance = linker.instantiate(&mut store, module)?;

        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| anyhow::anyhow!("Module has no exported 'memory'"))?;

        // Write input arrays into memory at offset 0
        let mut offset: usize = 0;
        let mut element_count: usize = 0;
        for input in inputs {
            let bytes = input.to_bytes();
            let needed = offset + bytes.len();

            let current_size = memory.data_size(&store);
            if needed > current_size {
                let pages_needed = ((needed - current_size) + 65535) / 65536;
                memory.grow(&mut store, pages_needed as u64)?;
            }

            let mem_data = memory.data_mut(&mut store);
            mem_data[offset..offset + bytes.len()].copy_from_slice(&bytes);
            element_count = input.element_count();
            offset += bytes.len();
        }

        // Call the function with (data_ptr=0, element_count)
        let func = instance
            .get_func(&mut store, func_name)
            .ok_or_else(|| anyhow::anyhow!("Function '{func_name}' not exported"))?;
        let params = vec![Val::I32(0), Val::I32(element_count as i32)];
        let mut results = vec![Val::I32(0)];
        func.call(&mut store, &params, &mut results)?;

        let result_ptr = match results[0] {
            Val::I32(v) => v as usize,
            _ => anyhow::bail!("Expected i32 result pointer from GPU function"),
        };

        // Read result back from memory
        let result_byte_len = result_count * 4;
        let mem_data = memory.data(&store);
        if result_ptr + result_byte_len > mem_data.len() {
            anyhow::bail!(
                "Result at offset {} with size {} exceeds memory bounds {}",
                result_ptr,
                result_byte_len,
                mem_data.len()
            );
        }
        let result_bytes = &mem_data[result_ptr..result_ptr + result_byte_len];
        Ok(ArrayValue::from_bytes_f32(result_bytes, result_count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_vec_add_f32() {
        let runtime = WasmRuntime::new().unwrap();

        // Hand-crafted SIMD WAT: adds two f32x4 vectors
        let wat = r#"
(module
  (memory (export "memory") 1)
  (func (export "vec_add_f32") (param $ptr i32) (param $len i32) (result i32)
    (local $i i32)
    (local $byte_len i32)
    (local $b_off i32)
    (local $r_off i32)
    ;; byte_len = len * 4
    (local.set $byte_len (i32.mul (local.get $len) (i32.const 4)))
    ;; B starts after A
    (local.set $b_off (local.get $byte_len))
    ;; Result starts after B
    (local.set $r_off (i32.mul (local.get $byte_len) (i32.const 2)))
    ;; SIMD loop: process 4 floats (16 bytes) at a time
    (local.set $i (i32.const 0))
    (block $break
      (loop $loop
        (br_if $break (i32.ge_u (local.get $i) (local.get $byte_len)))
        (v128.store
          (i32.add (local.get $r_off) (local.get $i))
          (f32x4.add
            (v128.load (i32.add (local.get $ptr) (local.get $i)))
            (v128.load (i32.add (local.get $b_off) (local.get $i)))))
        (local.set $i (i32.add (local.get $i) (i32.const 16)))
        (br $loop)))
    (local.get $r_off)))
"#;

        let (module, _) = runtime.compile_wat(wat).expect("SIMD WAT should compile");

        let a = ArrayValue::F32Array(vec![1.0, 2.0, 3.0, 4.0]);
        let b = ArrayValue::F32Array(vec![5.0, 6.0, 7.0, 8.0]);

        let result = runtime.call_array(&module, "vec_add_f32", &[a, b], 4).unwrap();

        match &result {
            ArrayValue::F32Array(v) => {
                assert_eq!(v.len(), 4);
                assert!((v[0] - 6.0).abs() < 0.001);
                assert!((v[1] - 8.0).abs() < 0.001);
                assert!((v[2] - 10.0).abs() < 0.001);
                assert!((v[3] - 12.0).abs() < 0.001);
            }
            _ => panic!("Expected F32Array result"),
        }
    }

    #[test]
    fn test_simd_larger_array() {
        let runtime = WasmRuntime::new().unwrap();

        // Same WAT, but with 8 elements (2 SIMD iterations)
        let wat = r#"
(module
  (memory (export "memory") 1)
  (func (export "vec_add_f32") (param $ptr i32) (param $len i32) (result i32)
    (local $i i32)
    (local $byte_len i32)
    (local $b_off i32)
    (local $r_off i32)
    (local.set $byte_len (i32.mul (local.get $len) (i32.const 4)))
    (local.set $b_off (local.get $byte_len))
    (local.set $r_off (i32.mul (local.get $byte_len) (i32.const 2)))
    (local.set $i (i32.const 0))
    (block $break
      (loop $loop
        (br_if $break (i32.ge_u (local.get $i) (local.get $byte_len)))
        (v128.store
          (i32.add (local.get $r_off) (local.get $i))
          (f32x4.add
            (v128.load (i32.add (local.get $ptr) (local.get $i)))
            (v128.load (i32.add (local.get $b_off) (local.get $i)))))
        (local.set $i (i32.add (local.get $i) (i32.const 16)))
        (br $loop)))
    (local.get $r_off)))
"#;

        let (module, _) = runtime.compile_wat(wat).unwrap();

        let a = ArrayValue::F32Array(vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]);
        let b = ArrayValue::F32Array(vec![0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]);

        let result = runtime.call_array(&module, "vec_add_f32", &[a, b], 8).unwrap();
        match &result {
            ArrayValue::F32Array(v) => {
                assert_eq!(v, &[1.5, 2.5, 3.5, 4.5, 11.0, 21.0, 31.0, 41.0]);
            }
            _ => panic!("Expected F32Array result"),
        }
    }

    #[test]
    fn test_gpu_double_via_linker() {
        use crate::gpu::GpuContext;
        use std::sync::{Arc, Mutex};

        let runtime = WasmRuntime::new().unwrap();
        let gpu = Arc::new(Mutex::new(GpuContext::new().expect("GPU should be available")));

        // WAT module that imports gpu.* functions and dispatches a doubling shader
        let wat = r#"
(module
  (import "gpu" "alloc" (func $gpu_alloc (param i32) (result i32)))
  (import "gpu" "write_buffer" (func $gpu_write (param i32 i32 i32)))
  (import "gpu" "dispatch_shader" (func $gpu_dispatch (param i32 i32 i32 i32)))
  (import "gpu" "read_buffer" (func $gpu_read (param i32 i32 i32)))
  (memory (export "memory") 1)
  (data (i32.const 4096)
    "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\0a@compute @workgroup_size(64)\0afn main(@builtin(global_invocation_id) id: vec3<u32>) {\0a  if id.x < arrayLength(&data) {\0a    data[id.x] = data[id.x] * 2.0;\0a  }\0a}\0a")
  (func (export "gpu_double_f32") (param $ptr i32) (param $len i32) (result i32)
    (local $byte_len i32)
    (local $buf i32)
    (local $wg i32)
    ;; byte_len = len * 4
    (local.set $byte_len (i32.mul (local.get $len) (i32.const 4)))
    ;; Allocate GPU buffer
    (local.set $buf (call $gpu_alloc (local.get $byte_len)))
    ;; Upload data to GPU
    (call $gpu_write (local.get $buf) (local.get $ptr) (local.get $byte_len))
    ;; Compute workgroups = ceil(len / 64)
    (local.set $wg (i32.div_u (i32.add (local.get $len) (i32.const 63)) (i32.const 64)))
    ;; Dispatch shader (data segment at offset 4096, length 175 bytes)
    (call $gpu_dispatch (i32.const 4096) (i32.const 224) (local.get $buf) (local.get $wg))
    ;; Read results back
    (call $gpu_read (local.get $buf) (local.get $ptr) (local.get $byte_len))
    (local.get $ptr)))
"#;

        let (module, _) = runtime.compile_wat(wat).expect("GPU WAT should compile");
        let input = ArrayValue::F32Array(vec![1.0, 2.0, 3.0, 4.0]);

        let result = runtime.call_gpu(&module, "gpu_double_f32", &gpu, &[input], 4).unwrap();
        match &result {
            ArrayValue::F32Array(v) => {
                assert_eq!(v, &[2.0, 4.0, 6.0, 8.0]);
            }
            _ => panic!("Expected F32Array result"),
        }
    }
}
