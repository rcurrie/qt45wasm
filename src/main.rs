mod llm;
mod runtime;
mod store;
mod types;

use anyhow::Result;

use llm::LlmClient;
use runtime::WasmRuntime;
use store::FunctionStore;

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
    args: &[i32],
) -> Result<i32> {
    println!("\nRequest: {name} ({description}) args: {args:?}");

    // 1. Check the store for a cached function
    if let Some(func) = store.get(name)? {
        println!("  [cache] found '{name}' (calls: {})", func.call_count);

        let module = if let Some(ref binary) = func.wasm_binary {
            // Fast path: load pre-compiled binary
            runtime.load_cached(binary)?
        } else if let Some(ref source) = func.source_code {
            // Recompile from source
            let (module, bytes) = runtime.compile_wat(source)?;
            store.save(name, description, signature, "wat", source, &bytes)?;
            module
        } else {
            anyhow::bail!("Function '{name}' has no source or binary");
        };

        let result = runtime.call_i32(&module, name, args)?;
        store.record_call(name)?;
        println!("  [wasm] {result}");
        return Ok(result);
    }

    // 2. Generate via LLM
    println!("  [cache] miss — generating via LLM...");
    let wat = llm.generate_wat(name, description, signature, runtime)?;
    let (module, bytes) = runtime.compile_wat(&wat)?;

    // 3. Store for future reuse
    store.save(name, description, signature, "wat", &wat, &bytes)?;

    // 4. Execute
    let result = runtime.call_i32(&module, name, args)?;
    store.record_call(name)?;
    println!("  [wasm] {result}");
    Ok(result)
}

fn main() -> Result<()> {
    let store = FunctionStore::new(DB_PATH)?;
    let runtime = WasmRuntime::new()?;
    let llm = LlmClient::new(OLLAMA_URL, OLLAMA_MODEL);

    // Test 1: Generate "add" (first time hits LLM, second time uses cache)
    synthesize(
        &store, &runtime, &llm,
        "add", "adds two integers", "(i32, i32) -> i32",
        &[10, 20],
    )?;

    // Test 2: "add" again — should hit cache
    synthesize(
        &store, &runtime, &llm,
        "add", "adds two integers", "(i32, i32) -> i32",
        &[100, 200],
    )?;

    // Test 3: Generate "multiply"
    synthesize(
        &store, &runtime, &llm,
        "multiply", "multiplies two integers", "(i32, i32) -> i32",
        &[5, 5],
    )?;

    // Test 4: Something the mock couldn't do — subtract
    synthesize(
        &store, &runtime, &llm,
        "subtract", "subtracts the second integer from the first", "(i32, i32) -> i32",
        &[50, 8],
    )?;

    // Test 5: Something more interesting — max of two values
    synthesize(
        &store, &runtime, &llm,
        "max", "returns the larger of two integers", "(i32, i32) -> i32",
        &[42, 17],
    )?;

    // List all stored functions
    println!("\n--- Function Library ---");
    for f in store.list()? {
        println!(
            "  {:<12} {} [calls: {}, verified: {}]",
            f.name, f.signature, f.call_count, f.is_verified
        );
    }

    Ok(())
}
