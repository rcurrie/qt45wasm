//! LLM Model Evaluation Benchmark
//!
//! Tests each downloaded ollama model on its ability to generate correct
//! WAT code across all three compute tiers (scalar, SIMD, GPU).
//!
//! For each challenge we know the correct answer. The LLM generates code,
//! we compile it, run it, and compare output to the expected result.

use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

// ── LLM client (minimal, self-contained) ──────────────────────────────

const OLLAMA_URL: &str = "http://localhost:11434/v1/chat/completions";
const MAX_RETRIES: usize = 3;

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
}

#[derive(Serialize, Deserialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

fn chat(client: &Client, model: &str, messages: &[Message]) -> Result<String> {
    let req = ChatRequest {
        model: model.to_string(),
        messages: messages.to_vec(),
        temperature: 0.2,
    };
    let resp = client.post(OLLAMA_URL).json(&req).send()?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        anyhow::bail!("LLM API error (HTTP {status}): {body}");
    }
    let chat_resp: ChatResponse = resp.json()?;
    Ok(chat_resp
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .unwrap_or_default())
}

/// Extract WAT from LLM response (strips markdown fences, finds (module ...))
fn extract_wat(response: &str) -> String {
    let text = response.trim();
    // Strip markdown fences
    for prefix in &["```wat", "```wasm", "```"] {
        if let Some(rest) = text.strip_prefix(prefix) {
            if let Some(code) = rest.strip_suffix("```") {
                return code.trim().to_string();
            }
        }
    }
    // Find (module ...)
    if let Some(start) = text.find("(module") {
        let bytes = text.as_bytes();
        let mut depth = 0;
        let mut end = start;
        for (i, &b) in bytes[start..].iter().enumerate() {
            match b {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = start + i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        if end > start {
            return text[start..end].to_string();
        }
    }
    text.to_string()
}

// ── Challenge definitions ─────────────────────────────────────────────

#[derive(Clone)]
enum Tier {
    Scalar,
    Simd,
    Gpu,
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tier::Scalar => write!(f, "scalar"),
            Tier::Simd => write!(f, "simd"),
            Tier::Gpu => write!(f, "gpu"),
        }
    }
}

#[derive(Clone)]
struct Challenge {
    tier: Tier,
    name: String,
    description: String,
    /// For scalar: the signature string
    signature: Option<String>,
    /// For SIMD/GPU: the memory layout description
    memory_layout: Option<String>,
    /// Test cases: (inputs, expected_output)
    tests: Vec<TestCase>,
}

#[derive(Clone)]
enum TestCase {
    Scalar {
        args: Vec<wasmtime::Val>,
        expected: Vec<i64>, // i32 or i64 values for comparison
    },
    Array {
        inputs: Vec<Vec<f32>>,
        expected: Vec<f32>,
    },
}

fn build_challenges() -> Vec<Challenge> {
    vec![
        // ── Scalar challenges ─────────────────────────────
        Challenge {
            tier: Tier::Scalar,
            name: "add".into(),
            description: "adds two 32-bit integers".into(),
            signature: Some("(param i32 i32) (result i32)".into()),
            memory_layout: None,
            tests: vec![
                TestCase::Scalar {
                    args: vec![wasmtime::Val::I32(3), wasmtime::Val::I32(4)],
                    expected: vec![7],
                },
                TestCase::Scalar {
                    args: vec![wasmtime::Val::I32(-1), wasmtime::Val::I32(1)],
                    expected: vec![0],
                },
                TestCase::Scalar {
                    args: vec![wasmtime::Val::I32(0), wasmtime::Val::I32(0)],
                    expected: vec![0],
                },
            ],
        },
        Challenge {
            tier: Tier::Scalar,
            name: "multiply".into(),
            description: "multiplies two 32-bit integers".into(),
            signature: Some("(param i32 i32) (result i32)".into()),
            memory_layout: None,
            tests: vec![
                TestCase::Scalar {
                    args: vec![wasmtime::Val::I32(3), wasmtime::Val::I32(7)],
                    expected: vec![21],
                },
                TestCase::Scalar {
                    args: vec![wasmtime::Val::I32(-2), wasmtime::Val::I32(5)],
                    expected: vec![-10],
                },
                TestCase::Scalar {
                    args: vec![wasmtime::Val::I32(0), wasmtime::Val::I32(99)],
                    expected: vec![0],
                },
            ],
        },
        Challenge {
            tier: Tier::Scalar,
            name: "max_i32".into(),
            description: "returns the larger of two signed 32-bit integers".into(),
            signature: Some("(param i32 i32) (result i32)".into()),
            memory_layout: None,
            tests: vec![
                TestCase::Scalar {
                    args: vec![wasmtime::Val::I32(5), wasmtime::Val::I32(3)],
                    expected: vec![5],
                },
                TestCase::Scalar {
                    args: vec![wasmtime::Val::I32(-10), wasmtime::Val::I32(-3)],
                    expected: vec![-3],
                },
                TestCase::Scalar {
                    args: vec![wasmtime::Val::I32(7), wasmtime::Val::I32(7)],
                    expected: vec![7],
                },
            ],
        },
        // ── SIMD challenges ──────────────────────────────
        Challenge {
            tier: Tier::Simd,
            name: "vec_add_f32".into(),
            description: "element-wise addition of two f32 arrays".into(),
            signature: None,
            memory_layout: Some(
                "Input A: N f32 values at offset 0. Input B: N f32 values at offset N*4. Result stored at offset N*8."
                    .into(),
            ),
            tests: vec![
                TestCase::Array {
                    inputs: vec![
                        vec![1.0, 2.0, 3.0, 4.0],
                        vec![5.0, 6.0, 7.0, 8.0],
                    ],
                    expected: vec![6.0, 8.0, 10.0, 12.0],
                },
                TestCase::Array {
                    inputs: vec![
                        vec![0.0, 0.0, 0.0, 0.0],
                        vec![1.0, 1.0, 1.0, 1.0],
                    ],
                    expected: vec![1.0, 1.0, 1.0, 1.0],
                },
            ],
        },
        Challenge {
            tier: Tier::Simd,
            name: "vec_mul_f32".into(),
            description: "element-wise multiplication of two f32 arrays".into(),
            signature: None,
            memory_layout: Some(
                "Input A: N f32 values at offset 0. Input B: N f32 values at offset N*4. Result stored at offset N*8."
                    .into(),
            ),
            tests: vec![
                TestCase::Array {
                    inputs: vec![
                        vec![2.0, 3.0, 4.0, 5.0],
                        vec![10.0, 10.0, 10.0, 10.0],
                    ],
                    expected: vec![20.0, 30.0, 40.0, 50.0],
                },
                TestCase::Array {
                    inputs: vec![
                        vec![1.0, 2.0, 3.0, 4.0],
                        vec![0.0, 0.0, 0.0, 0.0],
                    ],
                    expected: vec![0.0, 0.0, 0.0, 0.0],
                },
            ],
        },
        // ── GPU challenges ───────────────────────────────
        Challenge {
            tier: Tier::Gpu,
            name: "gpu_double_f32".into(),
            description: "doubles every element of an f32 array on the GPU".into(),
            signature: None,
            memory_layout: Some(
                "Input: N f32 values at offset 0. Result overwrites input at offset 0."
                    .into(),
            ),
            tests: vec![
                TestCase::Array {
                    inputs: vec![vec![1.0, 2.0, 3.0, 4.0]],
                    expected: vec![2.0, 4.0, 6.0, 8.0],
                },
                TestCase::Array {
                    inputs: vec![vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]],
                    expected: vec![20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0],
                },
            ],
        },
        Challenge {
            tier: Tier::Gpu,
            name: "gpu_negate_f32".into(),
            description: "negates every element of an f32 array on the GPU".into(),
            signature: None,
            memory_layout: Some(
                "Input: N f32 values at offset 0. Result overwrites input at offset 0."
                    .into(),
            ),
            tests: vec![
                TestCase::Array {
                    inputs: vec![vec![1.0, -2.0, 3.0, -4.0]],
                    expected: vec![-1.0, 2.0, -3.0, 4.0],
                },
                TestCase::Array {
                    inputs: vec![vec![0.0, 100.0, -100.0, 0.5]],
                    expected: vec![0.0, -100.0, 100.0, -0.5],
                },
            ],
        },
    ]
}

// ── Prompt builders (match the project's existing prompts) ────────────

fn scalar_system_prompt(name: &str, signature: &str) -> String {
    format!(
        r#"You are a WebAssembly expert. Generate WAT (WebAssembly Text Format) code only.
Rules:
- Output ONLY the WAT module. No markdown fences, no explanation, no comments.
- The module must export exactly one function named "{name}".
- The function signature is: {signature}
- Start your response with (module and end with )
- NEVER use numeric type/function indices like (type 0 ...) or (func 0). Use inline types only.
- Use (func (export "name") (param ...) (result ...) ...) syntax with inline export.

Example of correct WAT for a function that adds two integers:
(module
  (func (export "add") (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add))"#
    )
}

fn scalar_user_prompt(name: &str, description: &str, signature: &str) -> String {
    format!(
        r#"Generate a WAT module with a function named "{name}" that: {description}
Signature: {signature}
Use inline types only. No (type ...) declarations. No numeric indices."#
    )
}

fn simd_system_prompt(name: &str) -> String {
    format!(
        r#"You are a WebAssembly SIMD expert. Generate WAT code using v128 (SIMD) types.
Rules:
- Output ONLY the WAT module. No markdown fences, no explanation, no comments.
- The module MUST export memory: (memory (export "memory") 1)
- The module must export exactly one function named "{name}".
- Input data is pre-loaded into linear memory by the host at offset 0.
- The function takes two i32 params: (param $ptr i32) (param $len i32) where ptr is the data pointer and len is the element count.
- The function returns one i32: the byte offset where results begin in memory.
- Use v128 types: v128.load, v128.store, f32x4.add, f32x4.mul, f32x4.sub, f32x4.neg, etc.
- Process data in chunks of 4 (f32x4) with a loop. Step the loop index by 16 bytes (4 floats * 4 bytes).
- NEVER use numeric type indices like (type 0). Use inline types only.
- Use (func (export "name") ...) syntax with inline export.
- All v128.load and v128.store addresses MUST be multiples of 16 bytes.
- Start your response with (module and end with )

Memory layout convention:
- For single-array operations: data at offset 0, result after the data.
- For two-array operations (e.g., add A + B): A at offset 0, B at offset len*4 bytes, result at offset len*8 bytes.
- byte_len = len * 4 (for f32 arrays).

Example of correct SIMD WAT for element-wise addition of two f32 arrays:
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
    (local.get $r_off)))"#
    )
}

fn simd_user_prompt(name: &str, description: &str, memory_layout: &str) -> String {
    format!(
        r#"Generate a WAT module with SIMD v128 operations for a function named "{name}".
Description: {description}
Memory layout: {memory_layout}
The element count is passed as the second i32 parameter.
Use v128.load, v128.store, and f32x4.* operations to process 4 floats per loop iteration.
The function returns the byte offset where results are stored in memory."#
    )
}

fn gpu_system_prompt(name: &str) -> String {
    format!(
        r#"You are a WebAssembly + WebGPU compute expert. Generate a WAT module that uses GPU compute via host functions.
Rules:
- Output ONLY the WAT module. No markdown fences, no explanation, no comments.
- The module MUST import these host functions from the "gpu" namespace:
  (import "gpu" "alloc" (func $gpu_alloc (param i32) (result i32)))
  (import "gpu" "write_buffer" (func $gpu_write (param i32 i32 i32)))
  (import "gpu" "dispatch_shader" (func $gpu_dispatch (param i32 i32 i32 i32)))
  (import "gpu" "read_buffer" (func $gpu_read (param i32 i32 i32)))
- The module MUST export memory: (memory (export "memory") 1)
- The module must export exactly one function named "{name}".
- Input data is pre-loaded into linear memory at offset 0 by the host.
- The function takes (param $ptr i32) (param $len i32) and returns (result i32) — the byte offset of the result.
- Store the WGSL compute shader as a data segment at offset 4096: (data (i32.const 4096) "...")
- Use \0a for newlines inside the WGSL string, NOT actual newlines.
- The WGSL shader MUST declare: @group(0) @binding(0) var<storage, read_write> data: array<f32>;
- Workgroup size should be 64. Compute workgroups = ceil(len / 64).
- The function pattern is: alloc buffer → write data from $ptr → dispatch shader → read results back to $ptr → return $ptr.
- CRITICAL: gpu_read MUST write results back to $ptr (the same offset as the input). The return value MUST be $ptr.
- NEVER use numeric type/function indices. Use named imports and inline types only.
- Start your response with (module and end with )

Example: doubling all elements of an f32 array via GPU:
(module
  (import "gpu" "alloc" (func $gpu_alloc (param i32) (result i32)))
  (import "gpu" "write_buffer" (func $gpu_write (param i32 i32 i32)))
  (import "gpu" "dispatch_shader" (func $gpu_dispatch (param i32 i32 i32 i32)))
  (import "gpu" "read_buffer" (func $gpu_read (param i32 i32 i32)))
  (memory (export "memory") 1)
  (data (i32.const 4096)
    "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\0a@compute @workgroup_size(64)\0afn main(@builtin(global_invocation_id) id: vec3<u32>) {{\0a  if id.x < arrayLength(&data) {{\0a    data[id.x] = data[id.x] * 2.0;\0a  }}\0a}}\0a")
  (func (export "gpu_double_f32") (param $ptr i32) (param $len i32) (result i32)
    (local $byte_len i32)
    (local $buf i32)
    (local $wg i32)
    (local.set $byte_len (i32.mul (local.get $len) (i32.const 4)))
    (local.set $buf (call $gpu_alloc (local.get $byte_len)))
    (call $gpu_write (local.get $buf) (local.get $ptr) (local.get $byte_len))
    (local.set $wg (i32.div_u (i32.add (local.get $len) (i32.const 63)) (i32.const 64)))
    (call $gpu_dispatch (i32.const 4096) (i32.const 224) (local.get $buf) (local.get $wg))
    (call $gpu_read (local.get $buf) (local.get $ptr) (local.get $byte_len))
    (local.get $ptr)))"#
    )
}

fn gpu_user_prompt(name: &str, description: &str, memory_layout: &str) -> String {
    format!(
        r#"Generate a WAT module that uses GPU compute for a function named "{name}".
Description: {description}
Memory layout: {memory_layout}
The element count is passed as the second i32 parameter.
Include a WGSL compute shader as a data segment at offset 4096.
Use \0a for newlines in the WGSL string.
The shader must use @group(0) @binding(0) var<storage, read_write> data: array<f32>;
Workgroup size: 64. Compute workgroups = ceil(len / 64).
IMPORTANT: Count the exact byte length of your WGSL string and use that value in the gpu_dispatch call."#
    )
}

fn retry_prompt(error: &str) -> String {
    format!("The previous WAT failed to compile. Fix the error and output ONLY the corrected WAT module.\nError: {error}")
}

// ── Execution engine ──────────────────────────────────────────────────

use qt45::gpu::GpuContext;
use qt45::runtime::WasmRuntime;
use qt45::types::ArrayValue;

struct EvalResult {
    model: String,
    challenge: String,
    tier: Tier,
    compiled: bool,
    tests_passed: usize,
    tests_total: usize,
    attempts: usize,
    gen_time_ms: u64,
    error: Option<String>,
}

fn run_challenge(
    client: &Client,
    model: &str,
    challenge: &Challenge,
    runtime: &WasmRuntime,
    gpu: &Arc<Mutex<GpuContext>>,
) -> EvalResult {
    let start = Instant::now();

    // Build prompts
    let (system, user) = match &challenge.tier {
        Tier::Scalar => {
            let sig = challenge.signature.as_deref().unwrap();
            (
                scalar_system_prompt(&challenge.name, sig),
                scalar_user_prompt(&challenge.name, &challenge.description, sig),
            )
        }
        Tier::Simd => {
            let ml = challenge.memory_layout.as_deref().unwrap();
            (
                simd_system_prompt(&challenge.name),
                simd_user_prompt(&challenge.name, &challenge.description, ml),
            )
        }
        Tier::Gpu => {
            let ml = challenge.memory_layout.as_deref().unwrap();
            (
                gpu_system_prompt(&challenge.name),
                gpu_user_prompt(&challenge.name, &challenge.description, ml),
            )
        }
    };

    let mut messages = vec![
        Message {
            role: "system".into(),
            content: system,
        },
        Message {
            role: "user".into(),
            content: user,
        },
    ];

    // Try up to MAX_RETRIES with error feedback
    let mut last_error = String::new();
    for attempt in 1..=MAX_RETRIES {
        let response = match chat(client, model, &messages) {
            Ok(r) => r,
            Err(e) => {
                return EvalResult {
                    model: model.into(),
                    challenge: challenge.name.clone(),
                    tier: challenge.tier.clone(),
                    compiled: false,
                    tests_passed: 0,
                    tests_total: challenge.tests.len(),
                    attempts: attempt,
                    gen_time_ms: start.elapsed().as_millis() as u64,
                    error: Some(format!("LLM call failed: {e}")),
                };
            }
        };

        let wat = extract_wat(&response);

        // Try to compile
        match runtime.compile_wat(&wat) {
            Ok((module, _)) => {
                // Compiled! Run tests.
                let mut passed = 0;
                let total = challenge.tests.len();

                for tc in &challenge.tests {
                    // Wrap in catch_unwind — wgpu panics on invalid shaders
                    let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        match tc {
                            TestCase::Scalar { args, expected } => {
                                run_scalar_test(runtime, &module, &challenge.name, args, expected)
                            }
                            TestCase::Array { inputs, expected } => match &challenge.tier {
                                Tier::Simd => run_simd_test(
                                    runtime,
                                    &module,
                                    &challenge.name,
                                    inputs,
                                    expected,
                                ),
                                Tier::Gpu => run_gpu_test(
                                    runtime,
                                    &module,
                                    &challenge.name,
                                    gpu,
                                    inputs,
                                    expected,
                                ),
                                _ => false,
                            },
                        }
                    }))
                    .unwrap_or(false);
                    if ok {
                        passed += 1;
                    }
                }

                return EvalResult {
                    model: model.into(),
                    challenge: challenge.name.clone(),
                    tier: challenge.tier.clone(),
                    compiled: true,
                    tests_passed: passed,
                    tests_total: total,
                    attempts: attempt,
                    gen_time_ms: start.elapsed().as_millis() as u64,
                    error: None,
                };
            }
            Err(e) => {
                last_error = format!("{e:#}");
                if attempt < MAX_RETRIES {
                    // Feed error back for retry
                    messages.push(Message {
                        role: "assistant".into(),
                        content: response,
                    });
                    messages.push(Message {
                        role: "user".into(),
                        content: retry_prompt(&last_error),
                    });
                }
            }
        }
    }

    EvalResult {
        model: model.into(),
        challenge: challenge.name.clone(),
        tier: challenge.tier.clone(),
        compiled: false,
        tests_passed: 0,
        tests_total: challenge.tests.len(),
        attempts: MAX_RETRIES,
        gen_time_ms: start.elapsed().as_millis() as u64,
        error: Some(format!("compile failed: {last_error}")),
    }
}

fn run_scalar_test(
    runtime: &WasmRuntime,
    module: &wasmtime::Module,
    name: &str,
    args: &[wasmtime::Val],
    expected: &[i64],
) -> bool {
    let mut store = wasmtime::Store::new(runtime.engine(), ());
    let instance = match wasmtime::Instance::new(&mut store, module, &[]) {
        Ok(i) => i,
        Err(_) => return false,
    };
    let func = match instance.get_func(&mut store, name) {
        Some(f) => f,
        None => return false,
    };
    let mut results = vec![wasmtime::Val::I32(0); expected.len()];
    if func.call(&mut store, args, &mut results).is_err() {
        return false;
    }
    for (r, e) in results.iter().zip(expected.iter()) {
        let actual = match r {
            wasmtime::Val::I32(v) => *v as i64,
            wasmtime::Val::I64(v) => *v,
            _ => return false,
        };
        if actual != *e {
            return false;
        }
    }
    true
}

fn run_simd_test(
    runtime: &WasmRuntime,
    module: &wasmtime::Module,
    name: &str,
    inputs: &[Vec<f32>],
    expected: &[f32],
) -> bool {
    let array_inputs: Vec<ArrayValue> = inputs
        .iter()
        .map(|v| ArrayValue::F32Array(v.clone()))
        .collect();
    match runtime.call_array(module, name, &array_inputs, expected.len()) {
        Ok(ArrayValue::F32Array(v)) => {
            if v.len() != expected.len() {
                return false;
            }
            v.iter()
                .zip(expected.iter())
                .all(|(a, b)| (a - b).abs() < 0.001)
        }
        _ => false,
    }
}

fn run_gpu_test(
    runtime: &WasmRuntime,
    module: &wasmtime::Module,
    name: &str,
    gpu: &Arc<Mutex<GpuContext>>,
    inputs: &[Vec<f32>],
    expected: &[f32],
) -> bool {
    let array_inputs: Vec<ArrayValue> = inputs
        .iter()
        .map(|v| ArrayValue::F32Array(v.clone()))
        .collect();
    match runtime.call_gpu(module, name, gpu, &array_inputs, expected.len()) {
        Ok(ArrayValue::F32Array(v)) => {
            if v.len() != expected.len() {
                return false;
            }
            v.iter()
                .zip(expected.iter())
                .all(|(a, b)| (a - b).abs() < 0.001)
        }
        _ => false,
    }
}

// ── Main ──────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    // Suppress panic output from wgpu — we catch_unwind and report failures cleanly
    std::panic::set_hook(Box::new(|_| {}));

    let models = vec![
        "qwen2.5-coder:7b",
        "qwen3:8b",
        "deepseek-coder-v2:16b-lite-instruct-q4_K_M",
        "gemma2:27b",
        "glm-4.7-flash:q4_K_M",
    ];

    // Skip base (non-instruct) models — they don't follow chat format
    // "deepseek-coder-v2:16b-lite-base-q4_K_S" excluded

    let challenges = build_challenges();
    let runtime = WasmRuntime::new()?;
    let gpu = Arc::new(Mutex::new(GpuContext::new()?));
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?;

    println!("=== qt45 LLM Model Evaluation ===\n");
    println!(
        "Models: {}\nChallenges: {} ({} scalar, {} simd, {} gpu)\n",
        models.len(),
        challenges.len(),
        challenges.iter().filter(|c| matches!(c.tier, Tier::Scalar)).count(),
        challenges.iter().filter(|c| matches!(c.tier, Tier::Simd)).count(),
        challenges.iter().filter(|c| matches!(c.tier, Tier::Gpu)).count(),
    );

    let mut all_results: Vec<EvalResult> = Vec::new();

    for model in &models {
        println!("── {} ──", model);
        for challenge in &challenges {
            print!("  {:<20} [{:<6}] ", challenge.name, challenge.tier.to_string());
            std::io::stdout().flush().ok();

            let result = run_challenge(&client, model, challenge, &runtime, &gpu);

            if !result.compiled {
                println!(
                    "FAIL  compile failed ({} attempts, {}ms)",
                    result.attempts, result.gen_time_ms
                );
            } else if result.tests_passed == result.tests_total {
                println!(
                    "PASS  {}/{} tests ({} attempt{}, {}ms)",
                    result.tests_passed,
                    result.tests_total,
                    result.attempts,
                    if result.attempts > 1 { "s" } else { "" },
                    result.gen_time_ms
                );
            } else {
                println!(
                    "PARTIAL  {}/{} tests ({} attempt{}, {}ms)",
                    result.tests_passed,
                    result.tests_total,
                    result.attempts,
                    if result.attempts > 1 { "s" } else { "" },
                    result.gen_time_ms
                );
            }

            all_results.push(result);
        }
        println!();
    }

    // ── Summary table ─────────────────────────────────────────────────
    println!("\n=== SUMMARY ===\n");
    println!(
        "{:<45} {:>8} {:>8} {:>8} {:>10}",
        "MODEL", "COMPILE", "CORRECT", "SCORE", "TIME"
    );
    println!("{}", "-".repeat(83));

    for model in &models {
        let model_results: Vec<&EvalResult> =
            all_results.iter().filter(|r| r.model == *model).collect();
        let total = model_results.len();
        let compiled = model_results.iter().filter(|r| r.compiled).count();
        let fully_correct = model_results
            .iter()
            .filter(|r| r.compiled && r.tests_passed == r.tests_total)
            .count();
        let total_time: u64 = model_results.iter().map(|r| r.gen_time_ms).sum();

        println!(
            "{:<45} {:>5}/{:<2} {:>5}/{:<2} {:>7.0}% {:>8.1}s",
            model,
            compiled,
            total,
            fully_correct,
            total,
            (fully_correct as f64 / total as f64) * 100.0,
            total_time as f64 / 1000.0
        );
    }

    // Per-tier breakdown
    for tier_name in &["scalar", "simd", "gpu"] {
        println!("\n  {} tier:", tier_name.to_uppercase());
        println!(
            "  {:<45} {:>8} {:>8}",
            "MODEL", "COMPILE", "CORRECT"
        );
        println!("  {}", "-".repeat(65));

        for model in &models {
            let tier_results: Vec<&EvalResult> = all_results
                .iter()
                .filter(|r| r.model == *model && r.tier.to_string() == *tier_name)
                .collect();
            let total = tier_results.len();
            if total == 0 {
                continue;
            }
            let compiled = tier_results.iter().filter(|r| r.compiled).count();
            let correct = tier_results
                .iter()
                .filter(|r| r.compiled && r.tests_passed == r.tests_total)
                .count();

            println!(
                "  {:<45} {:>5}/{:<2} {:>5}/{:<2}",
                model, compiled, total, correct, total
            );
        }
    }

    println!();
    Ok(())
}
