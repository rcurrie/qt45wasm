# QT45

A self-improving agentic loop inspired by [QT45](https://www.science.org/doi/10.1126/science.adu3023), a 45-nucleotide RNA ribozyme capable of self-replication. An LLM synthesizes pure functions as WebAssembly modules, stores them in a persistent library, and reuses them to solve future problems. Each interaction makes the system more capable. The compute migrates from the probabilistic, expensive layer (LLM) to the deterministic, cheap layer (WASM) as the library grows. Just as QT45 catalyzes RNA-templated synthesis from a minimal motif - building complexity from simplicity in eutectic ice - this system starts with a minimal kernel and accumulates capability over time.

## Biology → Code

| Biology         | Code                 | Role                                                     |
| --------------- | -------------------- | -------------------------------------------------------- |
| RNA Polymerase  | `LlmClient`          | Synthesizes new functions from descriptions              |
| Ribosome        | `WasmRuntime`        | Compiles and executes WASM modules                       |
| DNA Helix       | `FunctionStore`      | Persistent memory of all known functions                 |
| Immune memory   | `HybridSearch`       | Recognizes previously seen problems (FTS5 + embeddings)  |
| Nucleotide pool | Prompt templates     | Raw material the polymerase works from                   |
| Enzyme          | Compiled WASM binary | Fast, deterministic, reusable computation                |
| Eutectic ice    | SQLite               | The substrate that concentrates and preserves everything |

## Architecture

```
┌─────────────────────────────────────────────┐
│                   REPL                      │
│  User input → Search → Plan → Execute → Out │
└────────┬──────────┬──────────┬──────────────┘
         │          │          │
    ┌────▼───┐ ┌────▼───┐ ┌────▼───────┐
    │ Search │ │  LLM   │ │  Runtime   │
    │ (FTS5 +│ │ Client │ │ (wasmtime) │
    │ vector)│ │        │ │            │
    └────┬───┘ └────┬───┘ └────┬───────┘
         │          │          │
    ┌────▼──────────▼──────────▼───┐
    │       FunctionStore          │
    │   (SQLite: functions, tests, │
    │    embeddings, agent state)  │
    └──────────────────────────────┘
```

## Install

Requires [Rust](https://rustup.rs/) and [Ollama](https://ollama.com/).

```bash
# Install a code-capable model
ollama pull qwen2.5-coder:7b

# Clone and build
git clone https://github.com/rcurrie/qt45.git qt45
cd qt45
cargo build
```

## Run

Make sure Ollama is running, then:

```bash
cargo run
```

First run generates functions via the LLM, verifies them with auto-generated tests, and caches everything in qt45.db. To reset, delete qt45.db.

```
Request: add (adds two integers) args: [I32(10), I32(20)]
  [cache] miss — generating via LLM...
  [llm] attempt 1/3...
  [llm] compilation successful
  [store] saved function: 'add'
  [test] generating test cases...
  [test] 3/3 passed — verified
  [wasm] 30

Request: add (adds two integers) args: [I32(100), I32(200)]
  [cache] found 'add' (calls: 1, verified)
  [wasm] 300

Request: multiply (multiplies two integers) args: [I32(5), I32(5)]
  [cache] miss — generating via LLM...
  [llm] attempt 1/3...
  [llm] compilation successful
  [store] saved function: 'multiply'
  [test] generating test cases...
  [test] 3/3 passed — verified
  [wasm] 25

Request: negate (returns the negation of an integer) args: [I32(42)]
  [cache] miss — generating via LLM...
  [llm] attempt 1/3...
  [llm] compilation successful
  [store] saved function: 'negate'
  [test] generating test cases...
  [test] 3/3 passed — verified
  [wasm] -42

Request: circle_area (returns the area of a circle given its radius (pi * r * r)) args: [F64(5.0)]
  [cache] miss — generating via LLM...
  [llm] attempt 1/3...
  [llm] compilation successful
  [store] saved function: 'circle_area'
  [test] generating test cases...
  [test] 3/3 passed — verified
  [wasm] 78.53981633974483

Request: sum (adds two numbers together) args: [I32(7), I32(8)]
  [search] matched 'add' via both (score: 0.0328)
  [wasm] 15

--- Function Library ---
  add            (i32, i32) -> i32    [calls: 3, verified: true]
  answer         () -> i32            [calls: 1, verified: true]
  circle_area    (f64) -> f64         [calls: 1, verified: true]
  max            (i32, i32) -> i32    [calls: 1, verified: true]
  multiply       (i32, i32) -> i32    [calls: 1, verified: true]
  negate         (i32) -> i32         [calls: 1, verified: true]
  subtract       (i32, i32) -> i32    [calls: 1, verified: true]
```

Key behaviors:
- **Generate**: new functions are synthesized as WAT, compiled to WASM, and verified with LLM-generated test cases
- **Cache**: subsequent calls to the same function load the compiled binary instantly
- **Search**: requesting "sum" finds the existing "add" function via hybrid FTS5 + vector similarity search, avoiding redundant LLM generation
