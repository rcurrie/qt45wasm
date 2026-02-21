# Findings

## LLM Model Evaluation for WAT Code Generation

We benchmarked five locally-hosted ollama models across all three compute tiers (scalar, SIMD, GPU) using 7 challenges with known-correct expected outputs. Each model gets up to 3 attempts per challenge with compilation error feedback. The benchmark lives in `src/bin/eval_models.rs` and can be re-run with `cargo run --bin eval_models`.

### Results

| Model | Size | Compile | Correct | Score | Time |
|---|---|---|---|---|---|
| **qwen2.5-coder:7b** | 4.7 GB | **7/7** | **6/7** | **86%** | **114s** |
| deepseek-coder-v2:16b-lite-instruct | 10 GB | 6/7 | 4/7 | 57% | 90s |
| gemma2:27b | 15 GB | 6/7 | 4/7 | 57% | 579s |
| qwen3:8b | 5.2 GB | 3/7 | 3/7 | 43% | 1963s |
| glm-4.7-flash:q4_K_M | 19 GB | 4/7 | 3/7 | 43% | 1389s |

### Per-tier breakdown

**Scalar** (add, multiply, max_i32):

| Model | Compile | Correct |
|---|---|---|
| qwen2.5-coder:7b | 3/3 | 3/3 |
| qwen3:8b | 2/3 | 2/3 |
| deepseek-coder-v2:16b | 2/3 | 2/3 |
| gemma2:27b | 2/3 | 2/3 |
| glm-4.7-flash | 2/3 | 2/3 |

**SIMD** (vec_add_f32, vec_mul_f32):

| Model | Compile | Correct |
|---|---|---|
| qwen2.5-coder:7b | 2/2 | 2/2 |
| deepseek-coder-v2:16b | 2/2 | 2/2 |
| gemma2:27b | 2/2 | 2/2 |
| qwen3:8b | 1/2 | 1/2 |
| glm-4.7-flash | 1/2 | 1/2 |

**GPU** (gpu_double_f32, gpu_negate_f32):

| Model | Compile | Correct |
|---|---|---|
| qwen2.5-coder:7b | 2/2 | 1/2 |
| deepseek-coder-v2:16b | 2/2 | 0/2 |
| gemma2:27b | 2/2 | 0/2 |
| qwen3:8b | 0/2 | 0/2 |
| glm-4.7-flash | 1/2 | 0/2 |

### Analysis

**Scalar tier** — All models handle simple arithmetic (add, multiply). Every model except qwen2.5-coder fails `max_i32`, which requires a conditional select rather than a single opcode. This is a good litmus test: models that can't emit `i32.gt_s` + `select` struggle with anything beyond trivial WAT.

**SIMD tier** — The code-specialized models (qwen2.5-coder, deepseek-coder-v2) and the large general model (gemma2:27b) all produce correct v128 SIMD loops. The non-code-specialized models (qwen3, glm) fail to compile valid SIMD WAT even with retry feedback.

**GPU tier** — This is the differentiator. qwen2.5-coder is the **only model to produce a working GPU shader** (gpu_double_f32 passed). The common failure mode across all models is getting the WGSL shader byte length wrong in the `gpu_dispatch` call — the WAT compiles fine but at runtime the shader string is truncated or includes trailing garbage, causing a wgpu validation panic. Even qwen2.5-coder misses gpu_negate_f32 for this reason.

**Speed vs quality** — Bigger models are not better here. gemma2:27b (15 GB) scores the same as deepseek-coder-v2 (10 GB) but takes 6x longer. qwen3:8b is the slowest overall due to repeated 300s timeouts on GPU challenges. qwen2.5-coder:7b is both the fastest and the most accurate — code-specific training matters more than parameter count for this task.

### Conclusion

The current default (`qwen2.5-coder:7b`) is the best available local model for this workload. The GPU tier remains the hard problem — no model reliably counts WGSL shader byte lengths. Potential mitigations: compute the shader length at the WAT level (host-side fixup), or add a post-generation validation step that checks the data segment length against the dispatch call argument.
