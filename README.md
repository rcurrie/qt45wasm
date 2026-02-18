# qt45wasm

A self-improving agentic loop inspired by [QT45](https://www.science.org/doi/10.1126/science.adu3023), a 45-nucleotide RNA ribozyme capable of self-replication. An LLM synthesizes pure functions as WebAssembly modules, stores them in a persistent library, and reuses them to solve future problems. Each interaction makes the system more capable. The compute migrates from the probabilistic, expensive layer (LLM) to the deterministic, cheap layer (WASM) as the library grows. Just as QT45 catalyzes RNA-templated synthesis from a minimal motif — building complexity from simplicity in eutectic ice — this system starts with a minimal kernel and accumulates capability over time.

## Compute Tiers

Functions climb a staircase of compute power as the workload demands:

```
  Scalar          SIMD (v128)         WebGPU
  ───────         ──────────          ──────
  i32/f64 in,     f32 arrays in       f32 arrays in
  scalar out      linear memory,      linear memory,
                  4 floats/op         Metal compute shader
                  (NEON on Apple      (thousands of
                  Silicon)            parallel threads)

  add(10, 20)     vec_add_f32         gpu_double_f32
       │          [1,2,3,4]           [1,2,3,4,5,6,7,8]
       ▼          [5,6,7,8]                  │
      30               │                     ▼
                       ▼            [2,4,6,8,10,12,14,16]
                  [6,8,10,12]
```

Each tier has its own code path — scalar functions are never touched when running SIMD or GPU workloads. The LLM generates the right kind of module for each tier: plain WAT for scalar, v128-enabled WAT for SIMD, and WAT with embedded WGSL compute shaders for GPU.

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
| Mitochondria    | `GpuContext`         | Offloads heavy computation to the GPU                    |

## Architecture

```
┌─────────────────────────────────────────────────┐
│                     REPL                        │
│  User input → Search → Plan → Execute → Output  │
└────────┬──────────┬──────────┬──────────────────┘
         │          │          │
    ┌────▼───┐ ┌────▼───┐ ┌────▼──────────────┐
    │ Search │ │  LLM   │ │     Runtime       │
    │ (FTS5 +│ │ Client │ │ call() scalar     │
    │ vector)│ │        │ │ call_array() simd │
    │        │ │        │ │ call_gpu() gpu    │
    └────┬───┘ └────┬───┘ └────┬──────────┬───┘
         │          │          │          │
         │          │          │     ┌────▼────┐
         │          │          │     │   GPU   │
         │          │          │     │ (wgpu/  │
         │          │          │     │  Metal) │
         │          │          │     └────┬────┘
    ┌────▼──────────▼──────────▼──────────▼───┐
    │          FunctionStore                  │
    │   (SQLite: functions, tests, shaders,   │
    │    embeddings, compute tiers)           │
    └─────────────────────────────────────────┘
```

## Install

Requires [Rust](https://rustup.rs/) and [Ollama](https://ollama.com/). GPU tier requires a Metal-capable Mac (Apple Silicon or discrete AMD GPU); SIMD and scalar work on any platform.

```bash
# Install a code-capable model
ollama pull qwen2.5-coder:7b

# Clone and build
git clone https://github.com/rcurrie/qt45.git qt45
cd qt45
cargo build
```

## Test

```
cargo test
```

## Run

Make sure Ollama is running, then try the built-in demo. The first run synthesizes functions via the LLM across all three compute tiers, verifies each with auto-generated tests, and stores everything in `qt45.db`:

```bash
cargo run -- --demo
```

```
Request: add (adds two integers) args: [I32(10), I32(20)]
  [cache] miss — generating via LLM...
  [llm] attempt 1/3...
  [llm] compilation successful
  [store] saved function: 'add' (tier: scalar)
  [test] 3/3 passed — verified
  [wasm] 30

Request: sum (adds two numbers together) args: [I32(7), I32(8)]
  [search] matched 'add' via both (score: 0.0328)
  [wasm] 15

=== SIMD Demo ===

Request [simd]: vec_add_f32 (element-wise addition of two f32 arrays) inputs: 2
  [cache] miss — generating SIMD via LLM...
  [llm] simd attempt 1/3...
  [llm] simd compilation successful
  [store] saved function: 'vec_add_f32' (tier: simd)
  [simd] [11, 22, 33, 44]

=== GPU Demo ===
  [gpu] adapter: Apple M1 Pro

Request [gpu]: gpu_double_f32 (double all elements of an f32 array) inputs: 1
  [cache] miss — generating GPU via LLM...
  [llm] gpu attempt 1/3...
  [llm] gpu compilation successful
  [store] saved function: 'gpu_double_f32' (tier: gpu)
  [gpu] [2, 4, 6, 8, 10, 12, 14, 16]

--- Function Library ---
  add            (i32, i32) -> i32    [calls: 3, tier: scalar, verified: true]
  vec_add_f32    simd                 [calls: 2, tier: simd,   verified: true]
  gpu_double_f32 gpu                  [calls: 2, tier: gpu,    verified: true]
  ...
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
Request [simd]: vec_add_f32 ...
  [cache] found 'vec_add_f32' simd (calls: 2, verified)
  [simd] [11, 22, 33, 44]
...
Request [gpu]: gpu_double_f32 ...
  [cache] found 'gpu_double_f32' gpu (calls: 2, verified)
  [gpu] [2, 4, 6, 8, 10, 12, 14, 16]
```

### Interactive REPL

For interactive use, run without `--demo`:

```bash
cargo run
```

#### Scalar functions

```
qt45> add "adds two integers" (i32, i32) -> i32 : 10, 20

Request: add (adds two integers) args: [I32(10), I32(20)]
  [cache] found 'add' (calls: 9, verified)
  [wasm] 30
```

#### SIMD array functions

Prefix with `simd` and pass arrays in brackets:

```
qt45> simd vec_add_f32 "element-wise add" [1,2,3,4] [5,6,7,8]

Request [simd]: vec_add_f32 (element-wise add) inputs: 2
  [simd] [6, 8, 10, 12]
```

#### GPU array functions

Prefix with `gpu` — the GPU context initializes lazily on first use:

```
qt45> gpu gpu_double_f32 "double all elements" [1,2,3,4]
  [gpu] initializing GPU context...
  [gpu] adapter: Apple M1 Pro

Request [gpu]: gpu_double_f32 (double all elements) inputs: 1
  [gpu] [2, 4, 6, 8]
```

#### Commands

```
qt45> .list
NAME             SIGNATURE                CALLS  TIER     VERIFIED
--------------------------------------------------------------------
add              (i32, i32) -> i32           10  scalar   yes
vec_add_f32      simd                         2  simd     yes
gpu_double_f32   gpu                          2  gpu      yes

qt45> .source add
(module
  (func (export "add") (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add))

qt45> .shader gpu_double_f32
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if id.x < arrayLength(&data) {
    data[id.x] = data[id.x] * 2.0;
  }
}
```

| Command             | Description                                   |
| ------------------- | --------------------------------------------- |
| `.list`             | List all stored functions with compute tier   |
| `.info <name>`      | Show function details including tier          |
| `.source <name>`    | Print WAT source code                         |
| `.shader <name>`    | Print WGSL shader source (GPU functions)      |
| `.tests <name>`     | Show test cases                               |
| `.bench <name> <n>` | Benchmark a SIMD/GPU function with n elements |
| `.delete <name>`    | Delete a function                             |
| `.help`             | Show commands and syntax                      |
| `.quit`             | Exit                                          |

Use `--model` to select a different Ollama model:

```bash
cargo run -- --model qwen2.5-coder:7b
```

### Storage

All state persists locally in the project directory:

- `qt45.db` — functions, tests, shaders, and vector embeddings (SQLite)
- `cache/` — the [BGE-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) ONNX embedding model (~50MB, downloaded on first run)

To reset, delete `qt45.db`. To re-download the embedding model, delete `cache/`.

### Key Behaviors

- **Generate**: new functions are synthesized as WAT (scalar and SIMD) or WAT+WGSL (GPU), compiled to WASM, and verified with LLM-generated test cases
- **Cache**: subsequent calls to the same function load the compiled binary instantly — no LLM needed
- **Search**: requesting "sum" finds the existing "add" function via hybrid FTS5 + vector similarity search, avoiding redundant LLM generation
- **Tiered compute**: scalar functions use plain WASM values, SIMD functions operate on arrays in linear memory with v128 instructions, GPU functions dispatch WGSL compute shaders via Metal
- **GPU bridge**: GPU functions use a host-function import pattern — the WAT module calls `gpu.alloc`, `gpu.write_buffer`, `gpu.dispatch_shader`, and `gpu.read_buffer` which the Rust host implements via wgpu
