mod llm;
mod runtime;
mod search;
mod store;
mod types;

use anyhow::Result;

use llm::LlmClient;
use runtime::WasmRuntime;
use search::HybridSearch;
use store::FunctionStore;
use types::Value;

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

fn parse_model_arg() -> String {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--model" {
            if let Some(val) = args.get(i + 1) {
                return val.clone();
            }
            eprintln!("error: --model requires a value");
            std::process::exit(1);
        }
        if let Some(val) = args[i].strip_prefix("--model=") {
            return val.to_string();
        }
        i += 1;
    }
    DEFAULT_MODEL.to_string()
}

fn main() -> Result<()> {
    let model = parse_model_arg();
    println!("[config] model: {model}");

    let store = FunctionStore::new(DB_PATH)?;
    let runtime = WasmRuntime::new()?;
    let llm = LlmClient::new(OLLAMA_URL, &model);
    let mut search = HybridSearch::new(&store)?;

    // --- i32 tests ---

    synthesize(
        &store, &runtime, &llm, &mut search,
        "add", "adds two integers", "(i32, i32) -> i32",
        &[Value::I32(10), Value::I32(20)],
    )?;

    // "add" again — should hit cache
    synthesize(
        &store, &runtime, &llm, &mut search,
        "add", "adds two integers", "(i32, i32) -> i32",
        &[Value::I32(100), Value::I32(200)],
    )?;

    synthesize(
        &store, &runtime, &llm, &mut search,
        "multiply", "multiplies two integers", "(i32, i32) -> i32",
        &[Value::I32(5), Value::I32(5)],
    )?;

    synthesize(
        &store, &runtime, &llm, &mut search,
        "subtract", "subtracts the second integer from the first", "(i32, i32) -> i32",
        &[Value::I32(50), Value::I32(8)],
    )?;

    synthesize(
        &store, &runtime, &llm, &mut search,
        "max", "returns the larger of two integers", "(i32, i32) -> i32",
        &[Value::I32(42), Value::I32(17)],
    )?;

    // --- single-arg i32 ---

    synthesize(
        &store, &runtime, &llm, &mut search,
        "negate", "returns the negation of an integer", "(i32) -> i32",
        &[Value::I32(42)],
    )?;

    // --- f64 ---

    synthesize(
        &store, &runtime, &llm, &mut search,
        "circle_area", "returns the area of a circle given its radius (pi * r * r)", "(f64) -> f64",
        &[Value::F64(5.0)],
    )?;

    // --- no args ---

    synthesize(
        &store, &runtime, &llm, &mut search,
        "answer", "returns the integer 42", "() -> i32",
        &[],
    )?;

    // --- search test: "sum" should find existing "add" ---

    synthesize(
        &store, &runtime, &llm, &mut search,
        "sum", "adds two numbers together", "(i32, i32) -> i32",
        &[Value::I32(7), Value::I32(8)],
    )?;

    // List all stored functions
    println!("\n--- Function Library ---");
    for f in store.list()? {
        println!(
            "  {:<14} {:<20} [calls: {}, verified: {}]",
            f.name, f.signature, f.call_count, f.is_verified
        );
    }

    Ok(())
}
