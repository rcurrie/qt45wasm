mod bench;
mod gpu;
mod llm;
mod runtime;
mod search;
mod store;
mod types;

use std::io::Write;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};

use gpu::GpuContext;
use llm::LlmClient;
use runtime::WasmRuntime;
use search::HybridSearch;
use store::FunctionStore;
use types::{ArrayValue, ComputeTier, Value};

const DB_PATH: &str = "qt45.db";
const OLLAMA_URL: &str = "http://localhost:11434/v1/chat/completions";
const DEFAULT_MODEL: &str = "qwen2.5-coder:7b";

/// The core loop: given a function request, retrieve from cache or generate via LLM,
/// then compile and execute.
fn synthesize(
    store: &FunctionStore,
    runtime: &WasmRuntime,
    llm: &LlmClient,
    search: &mut HybridSearch,
    name: &str,
    description: &str,
    signature: &str,
    args: &[Value],
) -> Result<Vec<Value>> {
    println!("\nRequest: {name} ({description}) args: {args:?}");

    // 1. Check the store for a cached function (exact name match)
    if let Some(func) = store.get(name)? {
        let verified = if func.is_verified { "verified" } else { "unverified" };
        println!("  [cache] found '{name}' (calls: {}, {verified})", func.call_count);

        let module = if let Some(ref binary) = func.wasm_binary {
            runtime.load_cached(binary)?
        } else if let Some(ref source) = func.source_code {
            let (module, bytes) = runtime.compile_wat(source)?;
            store.save(name, description, signature, "wat", source, &bytes)?;
            module
        } else {
            anyhow::bail!("Function '{name}' has no source or binary");
        };

        let results = runtime.call(&module, name, args)?;
        store.record_call(name)?;
        println!("  [wasm] {}", format_results(&results));
        return Ok(results);
    }

    // 1b. Search for a semantically similar existing function
    if let Some(hit) = search.find(store, name, description, signature)? {
        println!(
            "  [search] matched '{}' via {} (score: {:.4})",
            hit.function.name, hit.match_source, hit.score
        );

        let module = if let Some(ref binary) = hit.function.wasm_binary {
            runtime.load_cached(binary)?
        } else if let Some(ref source) = hit.function.source_code {
            runtime.compile_wat(source)?.0
        } else {
            anyhow::bail!("Matched function '{}' has no source or binary", hit.function.name);
        };

        let results = runtime.call(&module, &hit.function.name, args)?;
        store.record_call(&hit.function.name)?;
        println!("  [wasm] {}", format_results(&results));
        return Ok(results);
    }

    // 2. Generate via LLM
    println!("  [cache] miss — generating via LLM...");
    let wat = llm.generate_wat(name, description, signature, runtime)?;
    let (module, bytes) = runtime.compile_wat(&wat)?;

    // 3. Store for future reuse
    store.save(name, description, signature, "wat", &wat, &bytes)?;

    // 3b. Generate embedding for the new function
    if let Err(e) = search.embed_function(store, name, description) {
        println!("  [search] warning: failed to embed function: {e}");
    }

    // 4. Generate and run tests
    verify_function(store, runtime, llm, name, description, signature, &module);

    // 5. Execute
    let results = runtime.call(&module, name, args)?;
    store.record_call(name)?;
    println!("  [wasm] {}", format_results(&results));
    Ok(results)
}

/// SIMD synthesis: generate, store, verify, and execute an array-processing function.
fn synthesize_array(
    store: &FunctionStore,
    runtime: &WasmRuntime,
    llm: &LlmClient,
    search: &mut HybridSearch,
    name: &str,
    description: &str,
    inputs: &[ArrayValue],
) -> Result<ArrayValue> {
    let result_count = inputs.first().map(|a| a.element_count()).unwrap_or(0);
    println!("\nRequest [simd]: {name} ({description}) inputs: {}", inputs.len());

    // 1. Check cache
    if let Some(func) = store.get(name)? {
        if func.compute_tier == ComputeTier::Simd {
            let verified = if func.is_verified { "verified" } else { "unverified" };
            println!("  [cache] found '{name}' simd (calls: {}, {verified})", func.call_count);

            let module = if let Some(ref binary) = func.wasm_binary {
                runtime.load_cached(binary)?
            } else if let Some(ref source) = func.source_code {
                let (module, bytes) = runtime.compile_wat(source)?;
                store.save_with_tier(name, description, "", "wat", source, &bytes, ComputeTier::Simd, None)?;
                module
            } else {
                anyhow::bail!("Function '{name}' has no source or binary");
            };

            let result = runtime.call_array(&module, name, inputs, result_count)?;
            store.record_call(name)?;
            println!("  [simd] {result}");
            return Ok(result);
        }
    }

    // 2. Generate via LLM
    println!("  [cache] miss — generating SIMD via LLM...");
    let memory_layout = build_memory_layout(inputs);
    let wat = llm.generate_simd_wat(name, description, &memory_layout, runtime)?;
    let (module, bytes) = runtime.compile_wat(&wat)?;

    // 3. Store
    store.save_with_tier(name, description, "simd", "wat", &wat, &bytes, ComputeTier::Simd, None)?;

    // 3b. Embed
    if let Err(e) = search.embed_function(store, name, description) {
        println!("  [search] warning: failed to embed function: {e}");
    }

    // 4. Verify with LLM-generated array tests
    verify_array_function(store, runtime, llm, name, description, &module);

    // 5. Execute
    let result = runtime.call_array(&module, name, inputs, result_count)?;
    store.record_call(name)?;
    println!("  [simd] {result}");
    Ok(result)
}

/// Build a memory layout description for the LLM.
fn build_memory_layout(inputs: &[ArrayValue]) -> String {
    let mut parts = Vec::new();
    let labels = ["A", "B", "C", "D"];
    let mut offset_expr = String::new();
    for (i, input) in inputs.iter().enumerate() {
        let label = labels.get(i).unwrap_or(&"X");
        let count = input.element_count();
        if i == 0 {
            parts.push(format!("{label}: f32[N] at offset 0 (N={count})"));
            offset_expr = format!("{count}*4");
        } else {
            parts.push(format!("{label}: f32[N] at offset {offset_expr}"));
            offset_expr = format!("{}*4", count * (i + 1));
        }
    }
    parts.push(format!("Result: f32[N] at offset {offset_expr}"));
    parts.join(", ")
}

/// Verify an array function using LLM-generated test cases.
fn verify_array_function(
    store: &FunctionStore,
    runtime: &WasmRuntime,
    llm: &LlmClient,
    name: &str,
    description: &str,
    module: &wasmtime::Module,
) {
    println!("  [test] generating simd test cases...");
    let test_cases = llm.generate_simd_tests(name, description);
    if test_cases.is_empty() {
        println!("  [test] no simd test cases generated");
        return;
    }

    let mut passed = 0;
    let total = test_cases.len();

    for tc in &test_cases {
        let result_count = tc.expected.element_count();
        match runtime.call_array(module, name, &tc.inputs, result_count) {
            Ok(actual) => {
                if actual.approx_eq(&tc.expected) {
                    passed += 1;
                } else {
                    println!("  [test] FAIL: {name}(...) = {actual} (expected {})", tc.expected);
                }
            }
            Err(e) => {
                println!("  [test] ERROR: {name}(...) — {e}");
            }
        }
    }

    if passed == total {
        let _ = store.set_verified(name, true);
        println!("  [test] {passed}/{total} passed — verified");
    } else {
        println!("  [test] {passed}/{total} passed");
    }
}

/// GPU synthesis: generate, store, verify, and execute a GPU-accelerated function.
fn synthesize_gpu(
    store: &FunctionStore,
    runtime: &WasmRuntime,
    llm: &LlmClient,
    search: &mut HybridSearch,
    gpu: &Arc<Mutex<GpuContext>>,
    name: &str,
    description: &str,
    inputs: &[ArrayValue],
) -> Result<ArrayValue> {
    let result_count = inputs.first().map(|a| a.element_count()).unwrap_or(0);
    println!("\nRequest [gpu]: {name} ({description}) inputs: {}", inputs.len());

    // 1. Check cache
    if let Some(func) = store.get(name)? {
        if func.compute_tier == ComputeTier::Gpu {
            let verified = if func.is_verified { "verified" } else { "unverified" };
            println!("  [cache] found '{name}' gpu (calls: {}, {verified})", func.call_count);

            let module = if let Some(ref binary) = func.wasm_binary {
                runtime.load_cached(binary)?
            } else if let Some(ref source) = func.source_code {
                let (module, bytes) = runtime.compile_wat(source)?;
                store.save_with_tier(name, description, "", "wat", source, &bytes, ComputeTier::Gpu, func.shader_source.as_deref())?;
                module
            } else {
                anyhow::bail!("Function '{name}' has no source or binary");
            };

            let result = runtime.call_gpu(&module, name, gpu, inputs, result_count)?;
            store.record_call(name)?;
            println!("  [gpu] {result}");
            return Ok(result);
        }
    }

    // 2. Generate via LLM
    println!("  [cache] miss — generating GPU via LLM...");
    let memory_layout = build_memory_layout(inputs);
    let wat = llm.generate_gpu_wat(name, description, &memory_layout, runtime)?;
    let (module, bytes) = runtime.compile_wat(&wat)?;

    // Extract WGSL from the WAT data segment for storage
    let wgsl = extract_wgsl_from_wat(&wat);

    // 3. Store
    store.save_with_tier(name, description, "gpu", "wat", &wat, &bytes, ComputeTier::Gpu, wgsl.as_deref())?;

    // 3b. Embed
    if let Err(e) = search.embed_function(store, name, description) {
        println!("  [search] warning: failed to embed function: {e}");
    }

    // 4. Verify with LLM-generated tests (reuse SIMD tests — same array format)
    verify_gpu_function(store, runtime, llm, gpu, name, description, &module);

    // 5. Execute
    let result = runtime.call_gpu(&module, name, gpu, inputs, result_count)?;
    store.record_call(name)?;
    println!("  [gpu] {result}");
    Ok(result)
}

/// Extract the WGSL shader string from a WAT data segment at offset 4096.
fn extract_wgsl_from_wat(wat: &str) -> Option<String> {
    // Look for (data (i32.const 4096) "...")
    let marker = "(data (i32.const 4096)";
    let start = wat.find(marker)?;
    let after = &wat[start + marker.len()..];
    let quote_start = after.find('"')? + 1;
    let quote_end = after[quote_start..].find('"')?;
    let escaped = &after[quote_start..quote_start + quote_end];
    // Unescape \0a → newline
    Some(escaped.replace("\\0a", "\n"))
}

/// Verify a GPU function using LLM-generated test cases.
fn verify_gpu_function(
    store: &FunctionStore,
    runtime: &WasmRuntime,
    llm: &LlmClient,
    gpu: &Arc<Mutex<GpuContext>>,
    name: &str,
    description: &str,
    module: &wasmtime::Module,
) {
    println!("  [test] generating gpu test cases...");
    let test_cases = llm.generate_simd_tests(name, description);
    if test_cases.is_empty() {
        println!("  [test] no gpu test cases generated");
        return;
    }

    let mut passed = 0;
    let total = test_cases.len();

    for tc in &test_cases {
        let result_count = tc.expected.element_count();
        match runtime.call_gpu(module, name, gpu, &tc.inputs, result_count) {
            Ok(actual) => {
                if actual.approx_eq(&tc.expected) {
                    passed += 1;
                } else {
                    println!("  [test] FAIL: {name}(...) = {actual} (expected {})", tc.expected);
                }
            }
            Err(e) => {
                println!("  [test] ERROR: {name}(...) — {e}");
            }
        }
    }

    if passed == total {
        let _ = store.set_verified(name, true);
        println!("  [test] {passed}/{total} passed — verified");
    } else {
        println!("  [test] {passed}/{total} passed");
    }
}

/// Generate test cases via LLM, run them, and mark the function as verified if all pass.
/// Non-fatal: prints results but doesn't propagate errors.
fn verify_function(
    store: &FunctionStore,
    runtime: &WasmRuntime,
    llm: &LlmClient,
    name: &str,
    description: &str,
    signature: &str,
    module: &wasmtime::Module,
) {
    println!("  [test] generating test cases...");
    let test_cases = llm.generate_tests(name, description, signature);
    if test_cases.is_empty() {
        println!("  [test] no test cases generated");
        return;
    }

    let mut passed = 0;
    let total = test_cases.len();

    for tc in &test_cases {
        // Store the test case
        let test_id = match store.save_test(name, &tc.input, &tc.expected) {
            Ok(id) => id,
            Err(_) => continue,
        };

        // Run it
        match runtime.call(module, name, &tc.input) {
            Ok(actual) => {
                let ok = actual.len() == tc.expected.len()
                    && actual
                        .iter()
                        .zip(tc.expected.iter())
                        .all(|(a, e)| a.approx_eq(e));
                if ok {
                    passed += 1;
                    let _ = store.mark_test_passed(test_id);
                } else {
                    println!(
                        "  [test] FAIL: {}({}) = {} (expected {})",
                        name,
                        format_results(&tc.input),
                        format_results(&actual),
                        format_results(&tc.expected),
                    );
                }
            }
            Err(e) => {
                println!("  [test] ERROR: {}({}) — {e}", name, format_results(&tc.input));
            }
        }
    }

    if passed == total {
        let _ = store.set_verified(name, true);
        println!("  [test] {passed}/{total} passed — verified");
    } else {
        println!("  [test] {passed}/{total} passed");
    }
}

fn format_results(results: &[Value]) -> String {
    results
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

// --- CLI argument parsing ---

struct CliArgs {
    model: String,
    demo: bool,
}

fn parse_cli() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut model = DEFAULT_MODEL.to_string();
    let mut demo = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--demo" => demo = true,
            "--model" => {
                if let Some(val) = args.get(i + 1) {
                    model = val.clone();
                    i += 1;
                } else {
                    eprintln!("error: --model requires a value");
                    std::process::exit(1);
                }
            }
            other if other.starts_with("--model=") => {
                model = other.strip_prefix("--model=").unwrap().to_string();
            }
            other => {
                eprintln!("Unknown argument: {other}");
                eprintln!("Usage: qt45 [--model <name>] [--demo]");
                std::process::exit(1);
            }
        }
        i += 1;
    }
    CliArgs { model, demo }
}

// --- Input parsing ---

fn parse_call(input: &str) -> Result<(String, String, String, Vec<Value>)> {
    let input = input.trim();

    // 1. Extract name: first token
    let name_end = input
        .find(|c: char| c.is_whitespace())
        .ok_or_else(|| anyhow::anyhow!("Expected: name \"description\" (sig) -> type : args"))?;
    let name = &input[..name_end];

    // 2. Extract description: quoted string
    let rest = input[name_end..].trim_start();
    if !rest.starts_with('"') {
        anyhow::bail!("Expected quoted description after function name");
    }
    let desc_end = rest[1..]
        .find('"')
        .ok_or_else(|| anyhow::anyhow!("Unterminated description string"))?;
    let description = &rest[1..=desc_end];
    let rest = rest[desc_end + 2..].trim_start();

    // 3. Extract signature: everything up to ':' or end of line
    let (sig_part, args_part) = if let Some(colon_pos) = rest.find(':') {
        (&rest[..colon_pos], Some(rest[colon_pos + 1..].trim()))
    } else {
        (rest, None)
    };
    let signature = sig_part.trim().to_string();

    // 4. Parse parameter types from signature
    let param_types = parse_param_types(&signature)?;

    // 5. Parse arguments
    let args = if let Some(args_str) = args_part {
        parse_args(args_str, &param_types)?
    } else {
        if !param_types.is_empty() {
            anyhow::bail!(
                "Signature expects {} argument(s) — use : to provide them",
                param_types.len()
            );
        }
        Vec::new()
    };

    Ok((name.to_string(), description.to_string(), signature, args))
}

/// Parse a SIMD array call: name "description" [1,2,3,4] [5,6,7,8]
fn parse_array_call(input: &str) -> Result<(String, String, Vec<ArrayValue>)> {
    let input = input.trim();

    // 1. Extract name
    let name_end = input
        .find(|c: char| c.is_whitespace())
        .ok_or_else(|| anyhow::anyhow!("Expected: name \"description\" [array1] [array2]"))?;
    let name = &input[..name_end];

    // 2. Extract description
    let rest = input[name_end..].trim_start();
    if !rest.starts_with('"') {
        anyhow::bail!("Expected quoted description after function name");
    }
    let desc_end = rest[1..]
        .find('"')
        .ok_or_else(|| anyhow::anyhow!("Unterminated description string"))?;
    let description = &rest[1..=desc_end];
    let rest = rest[desc_end + 2..].trim_start();

    // 3. Parse arrays: each is [num, num, ...]
    let mut arrays = Vec::new();
    let mut remaining = rest;
    while let Some(start) = remaining.find('[') {
        let end = remaining[start..]
            .find(']')
            .ok_or_else(|| anyhow::anyhow!("Unterminated array bracket"))?;
        let array_str = &remaining[start + 1..start + end];
        let vals: Vec<f32> = array_str
            .split(',')
            .map(|s| {
                s.trim()
                    .parse::<f32>()
                    .context("Invalid f32 in array")
            })
            .collect::<Result<_>>()?;
        arrays.push(ArrayValue::F32Array(vals));
        remaining = &remaining[start + end + 1..];
    }

    if arrays.is_empty() {
        anyhow::bail!("Expected at least one array argument [...]");
    }

    Ok((name.to_string(), description.to_string(), arrays))
}

fn parse_param_types(signature: &str) -> Result<Vec<String>> {
    let open = signature
        .find('(')
        .ok_or_else(|| anyhow::anyhow!("Signature missing opening '('"))?;
    let close = signature
        .find(')')
        .ok_or_else(|| anyhow::anyhow!("Signature missing closing ')'"))?;
    let params_str = &signature[open + 1..close];
    if params_str.trim().is_empty() {
        return Ok(Vec::new());
    }
    Ok(params_str.split(',').map(|s| s.trim().to_string()).collect())
}

fn parse_args(args_str: &str, param_types: &[String]) -> Result<Vec<Value>> {
    let parts: Vec<&str> = args_str.split(',').map(|s| s.trim()).collect();
    if parts.len() != param_types.len() {
        anyhow::bail!(
            "Expected {} argument(s) but got {}",
            param_types.len(),
            parts.len()
        );
    }
    parts
        .iter()
        .zip(param_types.iter())
        .map(|(val_str, ty)| parse_value(val_str, ty))
        .collect()
}

fn parse_value(s: &str, ty: &str) -> Result<Value> {
    match ty {
        "i32" => Ok(Value::I32(s.parse::<i32>().context("Invalid i32 value")?)),
        "i64" => Ok(Value::I64(s.parse::<i64>().context("Invalid i64 value")?)),
        "f32" => Ok(Value::F32(s.parse::<f32>().context("Invalid f32 value")?)),
        "f64" => Ok(Value::F64(s.parse::<f64>().context("Invalid f64 value")?)),
        _ => anyhow::bail!("Unknown type in signature: '{ty}'"),
    }
}

// --- REPL command handlers ---

fn cmd_list(store: &FunctionStore) -> Result<()> {
    let functions = store.list()?;
    if functions.is_empty() {
        println!("No functions stored yet.");
        return Ok(());
    }
    println!(
        "{:<16} {:<24} {:>6}  {:<8} {}",
        "NAME", "SIGNATURE", "CALLS", "TIER", "VERIFIED"
    );
    println!("{}", "-".repeat(68));
    for f in &functions {
        let verified = if f.is_verified { "yes" } else { "no" };
        println!(
            "{:<16} {:<24} {:>6}  {:<8} {}",
            f.name, f.signature, f.call_count, f.compute_tier, verified
        );
    }
    Ok(())
}

fn cmd_info(store: &FunctionStore, name: &str) -> Result<()> {
    if name.is_empty() {
        anyhow::bail!("Usage: .info <function_name>");
    }
    match store.get(name)? {
        Some(f) => {
            println!("Name:        {}", f.name);
            println!("Description: {}", f.description);
            println!("Signature:   {}", f.signature);
            println!("Language:    {}", f.source_lang);
            println!("Tier:        {}", f.compute_tier);
            println!("Calls:       {}", f.call_count);
            println!("Verified:    {}", f.is_verified);
        }
        None => println!("Function '{name}' not found."),
    }
    Ok(())
}

fn cmd_source(store: &FunctionStore, name: &str) -> Result<()> {
    if name.is_empty() {
        anyhow::bail!("Usage: .source <function_name>");
    }
    match store.get(name)? {
        Some(f) => match f.source_code {
            Some(src) => println!("{src}"),
            None => println!("Function '{name}' has no source code stored."),
        },
        None => println!("Function '{name}' not found."),
    }
    Ok(())
}

fn cmd_tests(store: &FunctionStore, name: &str) -> Result<()> {
    if name.is_empty() {
        anyhow::bail!("Usage: .tests <function_name>");
    }
    let tests = store.get_tests(name)?;
    if tests.is_empty() {
        println!("No test cases for '{name}'.");
        return Ok(());
    }
    for tc in &tests {
        let input_str = format_results(&tc.input);
        let expected_str = format_results(&tc.expected);
        println!("  {name}({input_str}) = {expected_str}");
    }
    Ok(())
}

fn cmd_delete(store: &FunctionStore, name: &str) -> Result<()> {
    if name.is_empty() {
        anyhow::bail!("Usage: .delete <function_name>");
    }
    if store.get(name)?.is_none() {
        println!("Function '{name}' not found.");
        return Ok(());
    }
    print!("Delete function '{name}'? [y/N] ");
    std::io::stdout().flush()?;
    let mut confirm = String::new();
    std::io::stdin().read_line(&mut confirm)?;
    if confirm.trim().eq_ignore_ascii_case("y") {
        store.delete(name)?;
        println!("Deleted '{name}'.");
    } else {
        println!("Cancelled.");
    }
    Ok(())
}

fn cmd_bench(store: &FunctionStore, runtime: &WasmRuntime, gpu: &Option<Arc<Mutex<GpuContext>>>, arg: &str) -> Result<()> {
    let parts: Vec<&str> = arg.split_whitespace().collect();
    if parts.len() < 2 {
        anyhow::bail!("Usage: .bench <function_name> <size>");
    }
    let name = parts[0];
    let size: usize = parts[1].parse().context("Invalid size")?;
    let iterations = 100;

    let func = store
        .get(name)?
        .ok_or_else(|| anyhow::anyhow!("Function '{name}' not found"))?;

    let module = if let Some(ref binary) = func.wasm_binary {
        runtime.load_cached(binary)?
    } else if let Some(ref source) = func.source_code {
        runtime.compile_wat(source)?.0
    } else {
        anyhow::bail!("Function '{name}' has no source or binary");
    };

    // Generate test data (two arrays for binary ops)
    let a = bench::random_f32_array(size);
    let b = bench::random_f32_array(size);

    println!("\nBenchmark: {name} (size={size}, iterations={iterations})");

    match func.compute_tier {
        ComputeTier::Simd => {
            let result = bench::bench_array(runtime, &module, name, ComputeTier::Simd, &[a, b], size, iterations)?;
            bench::print_results(&[result]);
        }
        ComputeTier::Gpu => {
            let gpu = gpu.as_ref().ok_or_else(|| anyhow::anyhow!("GPU not initialized"))?;
            let result = bench::bench_gpu(runtime, &module, name, gpu, &[a, b], size, iterations)?;
            bench::print_results(&[result]);
        }
        ComputeTier::Scalar => {
            anyhow::bail!("Cannot benchmark scalar functions with .bench (use scalar call instead)");
        }
    }

    Ok(())
}

fn cmd_shader(store: &FunctionStore, name: &str) -> Result<()> {
    if name.is_empty() {
        anyhow::bail!("Usage: .shader <function_name>");
    }
    match store.get(name)? {
        Some(f) => match f.shader_source {
            Some(src) => println!("{src}"),
            None => println!("Function '{name}' has no WGSL shader stored."),
        },
        None => println!("Function '{name}' not found."),
    }
    Ok(())
}

fn cmd_help() {
    println!("COMMANDS:");
    println!("  .list              List all stored functions");
    println!("  .info <name>       Show function details");
    println!("  .source <name>     Print WAT source code");
    println!("  .shader <name>     Print WGSL shader source (GPU functions)");
    println!("  .tests <name>      Show test cases");
    println!("  .delete <name>     Delete a function");
    println!("  .bench <name> <n>  Benchmark a SIMD/GPU function with n elements");
    println!("  .help              Show this help");
    println!("  .quit / .exit      Exit the REPL");
    println!();
    println!("SCALAR FUNCTION CALLS:");
    println!("  name \"description\" (types) -> return_type : arg1, arg2");
    println!();
    println!("SIMD ARRAY CALLS:");
    println!("  simd name \"description\" [1,2,3,4] [5,6,7,8]");
    println!();
    println!("GPU ARRAY CALLS:");
    println!("  gpu name \"description\" [1,2,3,4] [5,6,7,8]");
    println!();
    println!("EXAMPLES:");
    println!("  add \"adds two integers\" (i32, i32) -> i32 : 10, 20");
    println!("  circle_area \"pi * r^2\" (f64) -> f64 : 5.0");
    println!("  answer \"returns 42\" () -> i32");
    println!("  simd vec_add_f32 \"element-wise add\" [1,2,3,4] [5,6,7,8]");
    println!("  gpu gpu_double_f32 \"double all elements\" [1,2,3,4]");
}

// --- REPL loop ---

fn handle_command(input: &str, store: &FunctionStore, runtime: &WasmRuntime, gpu: &Option<Arc<Mutex<GpuContext>>>) -> Result<()> {
    let parts: Vec<&str> = input.splitn(2, ' ').collect();
    let cmd = parts[0];
    let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

    match cmd {
        ".list" => cmd_list(store),
        ".info" => cmd_info(store, arg),
        ".source" => cmd_source(store, arg),
        ".shader" => cmd_shader(store, arg),
        ".tests" => cmd_tests(store, arg),
        ".delete" => cmd_delete(store, arg),
        ".bench" => cmd_bench(store, runtime, gpu, arg),
        ".help" => {
            cmd_help();
            Ok(())
        }
        ".quit" | ".exit" => Ok(()),
        _ => {
            println!("Unknown command: {cmd}. Type .help for available commands.");
            Ok(())
        }
    }
}

fn run_repl(
    store: &FunctionStore,
    runtime: &WasmRuntime,
    llm: &LlmClient,
    search: &mut HybridSearch,
) -> Result<()> {
    println!("qt45 — type .help for commands, .quit to exit\n");

    // Lazy GPU context — only initialized on first gpu command
    let mut gpu_ctx: Option<Arc<Mutex<GpuContext>>> = None;

    loop {
        print!("qt45> ");
        std::io::stdout().flush()?;

        let mut line = String::new();
        let bytes_read = std::io::stdin().read_line(&mut line)?;
        if bytes_read == 0 {
            println!();
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed.starts_with('.') {
            let should_quit = trimmed == ".quit" || trimmed == ".exit";
            if let Err(e) = handle_command(trimmed, store, runtime, &gpu_ctx) {
                eprintln!("Error: {e}");
            }
            if should_quit {
                break;
            }
        } else if trimmed.starts_with("simd ") {
            // SIMD array call
            match parse_array_call(&trimmed[5..]) {
                Ok((name, description, arrays)) => {
                    match synthesize_array(store, runtime, llm, search, &name, &description, &arrays) {
                        Ok(_) => {}
                        Err(e) => eprintln!("Error: {e:#}"),
                    }
                }
                Err(e) => {
                    eprintln!("Parse error: {e}");
                    eprintln!("Format: simd name \"description\" [1,2,3,4] [5,6,7,8]");
                }
            }
        } else if trimmed.starts_with("gpu ") {
            // Lazy GPU init
            if gpu_ctx.is_none() {
                println!("  [gpu] initializing GPU context...");
                match GpuContext::new() {
                    Ok(ctx) => {
                        println!("  [gpu] adapter: {}", ctx.adapter_name());
                        gpu_ctx = Some(Arc::new(Mutex::new(ctx)));
                    }
                    Err(e) => {
                        eprintln!("Error: failed to initialize GPU: {e}");
                        continue;
                    }
                }
            }
            let gpu = gpu_ctx.as_ref().unwrap();

            match parse_array_call(&trimmed[4..]) {
                Ok((name, description, arrays)) => {
                    match synthesize_gpu(store, runtime, llm, search, gpu, &name, &description, &arrays) {
                        Ok(_) => {}
                        Err(e) => eprintln!("Error: {e:#}"),
                    }
                }
                Err(e) => {
                    eprintln!("Parse error: {e}");
                    eprintln!("Format: gpu name \"description\" [1,2,3,4] [5,6,7,8]");
                }
            }
        } else {
            match parse_call(trimmed) {
                Ok((name, description, signature, args)) => {
                    match synthesize(store, runtime, llm, search, &name, &description, &signature, &args)
                    {
                        Ok(_) => {}
                        Err(e) => eprintln!("Error: {e:#}"),
                    }
                }
                Err(e) => {
                    eprintln!("Parse error: {e}");
                    eprintln!("Format: name \"description\" (types) -> type : arg1, arg2");
                }
            }
        }
    }

    Ok(())
}

// --- Demo mode (the original hardcoded test sequence) ---

fn run_demo(
    store: &FunctionStore,
    runtime: &WasmRuntime,
    llm: &LlmClient,
    search: &mut HybridSearch,
) -> Result<()> {
    synthesize(
        store, runtime, llm, search,
        "add", "adds two integers", "(i32, i32) -> i32",
        &[Value::I32(10), Value::I32(20)],
    )?;

    synthesize(
        store, runtime, llm, search,
        "add", "adds two integers", "(i32, i32) -> i32",
        &[Value::I32(100), Value::I32(200)],
    )?;

    synthesize(
        store, runtime, llm, search,
        "multiply", "multiplies two integers", "(i32, i32) -> i32",
        &[Value::I32(5), Value::I32(5)],
    )?;

    synthesize(
        store, runtime, llm, search,
        "subtract", "subtracts the second integer from the first", "(i32, i32) -> i32",
        &[Value::I32(50), Value::I32(8)],
    )?;

    synthesize(
        store, runtime, llm, search,
        "max", "returns the larger of two integers", "(i32, i32) -> i32",
        &[Value::I32(42), Value::I32(17)],
    )?;

    synthesize(
        store, runtime, llm, search,
        "negate", "returns the negation of an integer", "(i32) -> i32",
        &[Value::I32(42)],
    )?;

    synthesize(
        store, runtime, llm, search,
        "circle_area", "returns the area of a circle given its radius (pi * r * r)", "(f64) -> f64",
        &[Value::F64(5.0)],
    )?;

    synthesize(
        store, runtime, llm, search,
        "answer", "returns the integer 42", "() -> i32",
        &[],
    )?;

    synthesize(
        store, runtime, llm, search,
        "sum", "adds two numbers together", "(i32, i32) -> i32",
        &[Value::I32(7), Value::I32(8)],
    )?;

    // --- SIMD demo ---
    println!("\n=== SIMD Demo ===");
    synthesize_array(
        store, runtime, llm, search,
        "vec_add_f32", "element-wise addition of two f32 arrays",
        &[
            ArrayValue::F32Array(vec![1.0, 2.0, 3.0, 4.0]),
            ArrayValue::F32Array(vec![10.0, 20.0, 30.0, 40.0]),
        ],
    )?;

    synthesize_array(
        store, runtime, llm, search,
        "vec_add_f32", "element-wise addition of two f32 arrays",
        &[
            ArrayValue::F32Array(vec![100.0, 200.0, 300.0, 400.0]),
            ArrayValue::F32Array(vec![0.5, 0.5, 0.5, 0.5]),
        ],
    )?;

    // --- GPU demo ---
    println!("\n=== GPU Demo ===");
    println!("  [gpu] initializing GPU context...");
    let gpu = match GpuContext::new() {
        Ok(ctx) => {
            println!("  [gpu] adapter: {}", ctx.adapter_name());
            Arc::new(Mutex::new(ctx))
        }
        Err(e) => {
            eprintln!("  [gpu] skipping GPU demo — failed to initialize: {e}");
            println!("\n--- Function Library ---");
            for f in store.list()? {
                println!(
                    "  {:<14} {:<20} [calls: {}, tier: {}, verified: {}]",
                    f.name, f.signature, f.call_count, f.compute_tier, f.is_verified
                );
            }
            return Ok(());
        }
    };

    synthesize_gpu(
        store, runtime, llm, search, &gpu,
        "gpu_double_f32", "double all elements of an f32 array",
        &[ArrayValue::F32Array(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])],
    )?;

    synthesize_gpu(
        store, runtime, llm, search, &gpu,
        "gpu_double_f32", "double all elements of an f32 array",
        &[ArrayValue::F32Array(vec![10.0, 20.0, 30.0, 40.0])],
    )?;

    println!("\n--- Function Library ---");
    for f in store.list()? {
        println!(
            "  {:<14} {:<20} [calls: {}, tier: {}, verified: {}]",
            f.name, f.signature, f.call_count, f.compute_tier, f.is_verified
        );
    }

    Ok(())
}

// --- Entry point ---

fn main() -> Result<()> {
    let cli = parse_cli();
    println!("[config] model: {}", cli.model);

    let store = FunctionStore::new(DB_PATH)?;
    let runtime = WasmRuntime::new()?;
    let llm = LlmClient::new(OLLAMA_URL, &cli.model);
    let mut search = HybridSearch::new(&store)?;

    if cli.demo {
        run_demo(&store, &runtime, &llm, &mut search)
    } else {
        run_repl(&store, &runtime, &llm, &mut search)
    }
}

// --- Tests ---

#[cfg(test)]
mod repl_tests {
    use super::*;

    #[test]
    fn test_parse_call_basic() {
        let (name, desc, sig, args) =
            parse_call(r#"add "adds two integers" (i32, i32) -> i32 : 10, 20"#).unwrap();
        assert_eq!(name, "add");
        assert_eq!(desc, "adds two integers");
        assert_eq!(sig, "(i32, i32) -> i32");
        assert_eq!(args.len(), 2);
    }

    #[test]
    fn test_parse_call_no_args() {
        let (name, desc, sig, args) =
            parse_call(r#"answer "returns 42" () -> i32"#).unwrap();
        assert_eq!(name, "answer");
        assert_eq!(desc, "returns 42");
        assert_eq!(sig, "() -> i32");
        assert!(args.is_empty());
    }

    #[test]
    fn test_parse_call_f64() {
        let (_, _, _, args) =
            parse_call(r#"circle_area "pi * r^2" (f64) -> f64 : 5.0"#).unwrap();
        assert_eq!(args.len(), 1);
        match &args[0] {
            Value::F64(v) => assert!((v - 5.0).abs() < 0.001),
            _ => panic!("expected F64"),
        }
    }

    #[test]
    fn test_parse_call_negative() {
        let (_, _, _, args) =
            parse_call(r#"negate "negation" (i32) -> i32 : -42"#).unwrap();
        match &args[0] {
            Value::I32(v) => assert_eq!(*v, -42),
            _ => panic!("expected I32"),
        }
    }

    #[test]
    fn test_parse_param_types() {
        let types = parse_param_types("(i32, f64) -> i32").unwrap();
        assert_eq!(types, vec!["i32", "f64"]);
    }

    #[test]
    fn test_parse_param_types_empty() {
        let types = parse_param_types("() -> i32").unwrap();
        assert!(types.is_empty());
    }

    #[test]
    fn test_parse_call_missing_args_errors() {
        let result = parse_call(r#"add "adds two" (i32, i32) -> i32"#);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_array_call_basic() {
        let (name, desc, arrays) =
            parse_array_call(r#"vec_add_f32 "add two arrays" [1,2,3,4] [5,6,7,8]"#).unwrap();
        assert_eq!(name, "vec_add_f32");
        assert_eq!(desc, "add two arrays");
        assert_eq!(arrays.len(), 2);
        match &arrays[0] {
            ArrayValue::F32Array(v) => assert_eq!(v, &[1.0, 2.0, 3.0, 4.0]),
            _ => panic!("expected F32Array"),
        }
    }

    #[test]
    fn test_parse_array_call_single() {
        let (_, _, arrays) =
            parse_array_call(r#"vec_negate "negate array" [1.5, -2.5, 3.0, 4.0]"#).unwrap();
        assert_eq!(arrays.len(), 1);
    }
}
