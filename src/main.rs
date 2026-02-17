mod llm;
mod runtime;
mod store;
mod types;

use anyhow::Result;

use llm::LlmClient;
use runtime::WasmRuntime;
use store::FunctionStore;
use types::Value;

const DB_PATH: &str = "qt45.db";
const OLLAMA_URL: &str = "http://localhost:11434/v1/chat/completions";
const OLLAMA_MODEL: &str = "qwen2.5-coder:7b";

/// The core loop: given a function request, retrieve from cache or generate via LLM,
/// then compile and execute.
fn synthesize(
    store: &FunctionStore,
    runtime: &WasmRuntime,
    llm: &LlmClient,
    name: &str,
    description: &str,
    signature: &str,
    args: &[Value],
) -> Result<Vec<Value>> {
    println!("\nRequest: {name} ({description}) args: {args:?}");

    // 1. Check the store for a cached function
    if let Some(func) = store.get(name)? {
        println!("  [cache] found '{name}' (calls: {})", func.call_count);

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

    // 2. Generate via LLM
    println!("  [cache] miss — generating via LLM...");
    let wat = llm.generate_wat(name, description, signature, runtime)?;
    let (module, bytes) = runtime.compile_wat(&wat)?;

    // 3. Store for future reuse
    store.save(name, description, signature, "wat", &wat, &bytes)?;

    // 4. Execute
    let results = runtime.call(&module, name, args)?;
    store.record_call(name)?;
    println!("  [wasm] {}", format_results(&results));
    Ok(results)
}

fn format_results(results: &[Value]) -> String {
    results
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn main() -> Result<()> {
    let store = FunctionStore::new(DB_PATH)?;
    let runtime = WasmRuntime::new()?;
    let llm = LlmClient::new(OLLAMA_URL, OLLAMA_MODEL);

    // --- i32 tests (same as before) ---

    synthesize(
        &store, &runtime, &llm,
        "add", "adds two integers", "(i32, i32) -> i32",
        &[Value::I32(10), Value::I32(20)],
    )?;

    // "add" again — should hit cache
    synthesize(
        &store, &runtime, &llm,
        "add", "adds two integers", "(i32, i32) -> i32",
        &[Value::I32(100), Value::I32(200)],
    )?;

    synthesize(
        &store, &runtime, &llm,
        "multiply", "multiplies two integers", "(i32, i32) -> i32",
        &[Value::I32(5), Value::I32(5)],
    )?;

    synthesize(
        &store, &runtime, &llm,
        "subtract", "subtracts the second integer from the first", "(i32, i32) -> i32",
        &[Value::I32(50), Value::I32(8)],
    )?;

    synthesize(
        &store, &runtime, &llm,
        "max", "returns the larger of two integers", "(i32, i32) -> i32",
        &[Value::I32(42), Value::I32(17)],
    )?;

    // --- New: single-arg i32 ---

    synthesize(
        &store, &runtime, &llm,
        "negate", "returns the negation of an integer", "(i32) -> i32",
        &[Value::I32(42)],
    )?;

    // --- New: f64 ---

    synthesize(
        &store, &runtime, &llm,
        "circle_area", "returns the area of a circle given its radius (pi * r * r)", "(f64) -> f64",
        &[Value::F64(5.0)],
    )?;

    // --- New: no args ---

    synthesize(
        &store, &runtime, &llm,
        "answer", "returns the integer 42", "() -> i32",
        &[],
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
