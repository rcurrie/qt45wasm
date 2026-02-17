# qt45

A self-improving agentic loop inspired by [QT45](https://www.science.org/doi/10.1126/science.adu3023), a 45-nucleotide RNA ribozyme capable of self-replication. Just as QT45 catalyzes RNA-templated synthesis from a minimal motif — building complexity from simplicity in eutectic ice — this system starts with a minimal kernel and accumulates capability over time. An LLM synthesizes pure functions as WebAssembly modules, stores them in a persistent library, and reuses them to solve future problems. Each interaction potentially makes the system more capable. The compute migrates from the probabilistic, expensive layer (LLM) to the deterministic, cheap layer (WASM) as the library grows.

## Biology → Code

| Biology | Code | Role |
|---------|------|------|
| RNA Polymerase | `LlmClient` | Synthesizes new functions from descriptions |
| Ribosome | `WasmRuntime` | Compiles and executes WASM modules |
| DNA Helix | `FunctionStore` | Persistent memory of all known functions |
| Nucleotide pool | Prompt templates | Raw material the polymerase works from |
| Enzyme | Compiled WASM binary | Fast, deterministic, reusable computation |
| Eutectic ice | SQLite | The substrate that concentrates and preserves everything |

## Install

Requires [Rust](https://rustup.rs/) and [Ollama](https://ollama.com/).

```bash
# Install a code-capable model
ollama pull qwen2.5-coder:7b

# Clone and build
git clone <repo-url> qt45
cd qt45
cargo build
```

## Run

Make sure Ollama is running, then:

```bash
cargo run
```

First run generates functions via the LLM and caches them. Subsequent runs load from the database instantly.

```
Request: add (adds two integers) args: [10, 20]
  [cache] miss — generating via LLM...
  [llm] attempt 1/3...
  [llm] compilation successful
  [store] saved function: 'add'
  [wasm] 30

Request: add (adds two integers) args: [100, 200]
  [cache] found 'add' (calls: 1)
  [wasm] 300

--- Function Library ---
  add          (i32, i32) -> i32 [calls: 2, verified: false]
  max          (i32, i32) -> i32 [calls: 1, verified: false]
  multiply     (i32, i32) -> i32 [calls: 1, verified: false]
  subtract     (i32, i32) -> i32 [calls: 1, verified: false]
```
