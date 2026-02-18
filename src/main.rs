mod llm;
mod runtime;
mod search;
mod store;
mod types;

use std::io::Write;

use anyhow::{Context, Result};

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
        "{:<16} {:<24} {:>6}  {}",
        "NAME", "SIGNATURE", "CALLS", "VERIFIED"
    );
    println!("{}", "-".repeat(60));
    for f in &functions {
        let verified = if f.is_verified { "yes" } else { "no" };
        println!(
            "{:<16} {:<24} {:>6}  {}",
            f.name, f.signature, f.call_count, verified
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

fn cmd_help() {
    println!("COMMANDS:");
    println!("  .list              List all stored functions");
    println!("  .info <name>       Show function details");
    println!("  .source <name>     Print WAT source code");
    println!("  .tests <name>      Show test cases");
    println!("  .delete <name>     Delete a function");
    println!("  .help              Show this help");
    println!("  .quit / .exit      Exit the REPL");
    println!();
    println!("FUNCTION CALLS:");
    println!("  name \"description\" (types) -> return_type : arg1, arg2");
    println!();
    println!("EXAMPLES:");
    println!("  add \"adds two integers\" (i32, i32) -> i32 : 10, 20");
    println!("  circle_area \"pi * r^2\" (f64) -> f64 : 5.0");
    println!("  answer \"returns 42\" () -> i32");
}

// --- REPL loop ---

fn handle_command(input: &str, store: &FunctionStore) -> Result<()> {
    let parts: Vec<&str> = input.splitn(2, ' ').collect();
    let cmd = parts[0];
    let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

    match cmd {
        ".list" => cmd_list(store),
        ".info" => cmd_info(store, arg),
        ".source" => cmd_source(store, arg),
        ".tests" => cmd_tests(store, arg),
        ".delete" => cmd_delete(store, arg),
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
            if let Err(e) = handle_command(trimmed, store) {
                eprintln!("Error: {e}");
            }
            if should_quit {
                break;
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

    println!("\n--- Function Library ---");
    for f in store.list()? {
        println!(
            "  {:<14} {:<20} [calls: {}, verified: {}]",
            f.name, f.signature, f.call_count, f.is_verified
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
}
