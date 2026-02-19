# Findings

## GPU verification failures are LLM quality, not runtime bugs

The wasmtime + wgpu pipeline is solid — 1,500 calls with hand-crafted WAT, zero failures. All `verified: false` issues with `qwen2.5-coder:7b` trace to: wrong return pointers in generated WAT, wrong input counts in test cases, and wrong expected values. Scalar and SIMD tiers are reliable; GPU tier needs a more capable model.
