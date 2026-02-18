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

Make sure Ollama is running, then try the built-in demo. The first run synthesizes functions via the LLM, verifies each with auto-generated tests, and stores everything in `qt45.db`:

```bash
cargo run -- --demo
```

```
Request: add (adds two integers) args: [I32(10), I32(20)]
  [cache] miss — generating via LLM...
  [llm] attempt 1/3...
  [llm] compilation successful
  [store] saved function: 'add'
  [test] generating test cases...
  [test] 3/3 passed — verified
  [wasm] 30

Request: multiply (multiplies two integers) args: [I32(5), I32(5)]
  [cache] miss — generating via LLM...
  ...
  [wasm] 25

Request: sum (adds two numbers together) args: [I32(7), I32(8)]
  [search] matched 'add' via both (score: 0.0328)
  [wasm] 15

--- Function Library ---
  add            (i32, i32) -> i32    [calls: 3, verified: true]
  circle_area    (f64) -> f64         [calls: 1, verified: true]
  max            (i32, i32) -> i32    [calls: 1, verified: true]
  multiply       (i32, i32) -> i32    [calls: 1, verified: true]
  negate         (i32) -> i32         [calls: 1, verified: true]
  subtract       (i32, i32) -> i32    [calls: 1, verified: true]
```

Run it again — every function is now served from cache with zero LLM calls:

```bash
cargo run -- --demo
```

```
Request: add (adds two integers) args: [I32(10), I32(20)]
  [cache] found 'add' (calls: 3, verified)
  [wasm] 30
...
```

### Interactive REPL

For interactive use, run without `--demo`:

```bash
cargo run
```

```
qt45> add "adds two integers" (i32, i32) -> i32 : 10, 20

Request: add (adds two integers) args: [I32(10), I32(20)]
  [cache] found 'add' (calls: 9, verified)
  [wasm] 30

qt45> .list
NAME             SIGNATURE                 CALLS  VERIFIED
------------------------------------------------------------
add              (i32, i32) -> i32            10  yes
circle_area      (f64) -> f64                  2  yes
multiply         (i32, i32) -> i32             2  yes
...

qt45> .source add
(module
  (func (export "add") (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add))
```

| Command | Description |
|---------|-------------|
| `.list` | List all stored functions |
| `.info <name>` | Show function details |
| `.source <name>` | Print WAT source code |
| `.tests <name>` | Show test cases |
| `.delete <name>` | Delete a function |
| `.help` | Show commands and syntax |
| `.quit` | Exit |

Use `--model` to select a different Ollama model:

```bash
cargo run -- --model qwen2.5-coder:7b
```

### Storage

All state persists locally in the project directory:

- `qt45.db` — functions, tests, and vector embeddings (SQLite)
- `cache/` — the [BGE-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) ONNX embedding model (~50MB, downloaded on first run)

To reset, delete `qt45.db`. To re-download the embedding model, delete `cache/`.

### Key Behaviors

- **Generate**: new functions are synthesized as WAT, compiled to WASM, and verified with LLM-generated test cases
- **Cache**: subsequent calls to the same function load the compiled binary instantly
- **Search**: requesting "sum" finds the existing "add" function via hybrid FTS5 + vector similarity search, avoiding redundant LLM generation
