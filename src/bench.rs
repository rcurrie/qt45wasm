use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use wasmtime::Module;

use crate::gpu::GpuContext;
use crate::runtime::WasmRuntime;
use crate::types::{ArrayValue, ComputeTier};

/// Result of benchmarking a single function at a given tier.
pub struct BenchResult {
    pub tier: ComputeTier,
    pub element_count: usize,
    pub duration_us: u64,
    pub throughput_mflops: f64,
}

/// Benchmark an array function by running it `iterations` times.
pub fn bench_array(
    runtime: &WasmRuntime,
    module: &Module,
    func_name: &str,
    tier: ComputeTier,
    inputs: &[ArrayValue],
    result_count: usize,
    iterations: usize,
) -> Result<BenchResult> {
    let element_count = inputs.first().map(|a| a.element_count()).unwrap_or(0);

    // Warmup
    runtime.call_array(module, func_name, inputs, result_count)?;

    let start = Instant::now();
    for _ in 0..iterations {
        runtime.call_array(module, func_name, inputs, result_count)?;
    }
    let elapsed = start.elapsed();
    let duration_us = elapsed.as_micros() as u64;

    // FLOP estimate: element_count ops per iteration
    let total_ops = element_count as f64 * iterations as f64;
    let throughput_mflops = if duration_us > 0 {
        total_ops / (duration_us as f64)
    } else {
        0.0
    };

    Ok(BenchResult {
        tier,
        element_count,
        duration_us,
        throughput_mflops,
    })
}

/// Benchmark a GPU function by running it `iterations` times.
pub fn bench_gpu(
    runtime: &WasmRuntime,
    module: &Module,
    func_name: &str,
    gpu: &Arc<Mutex<GpuContext>>,
    inputs: &[ArrayValue],
    result_count: usize,
    iterations: usize,
) -> Result<BenchResult> {
    let element_count = inputs.first().map(|a| a.element_count()).unwrap_or(0);

    // Warmup
    runtime.call_gpu(module, func_name, gpu, inputs, result_count)?;

    let start = Instant::now();
    for _ in 0..iterations {
        runtime.call_gpu(module, func_name, gpu, inputs, result_count)?;
    }
    let elapsed = start.elapsed();
    let duration_us = elapsed.as_micros() as u64;

    let total_ops = element_count as f64 * iterations as f64;
    let throughput_mflops = if duration_us > 0 {
        total_ops / (duration_us as f64)
    } else {
        0.0
    };

    Ok(BenchResult {
        tier: ComputeTier::Gpu,
        element_count,
        duration_us,
        throughput_mflops,
    })
}

/// Generate random f32 test data of the given size (padded to multiple of 4).
pub fn random_f32_array(size: usize) -> ArrayValue {
    let padded = (size + 3) & !3;
    let mut v = Vec::with_capacity(padded);
    // Simple deterministic pseudo-random (not cryptographic, just for benchmarks)
    let mut seed: u32 = 12345;
    for _ in 0..padded {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let val = (seed >> 16) as f32 / 65536.0;
        v.push(val);
    }
    ArrayValue::F32Array(v)
}

/// Print benchmark results in a formatted table.
pub fn print_results(results: &[BenchResult]) {
    println!("  {:<8} {:>10} {:>12} {:>14}", "TIER", "ELEMENTS", "TIME (us)", "MFLOP/s");
    println!("  {}", "-".repeat(48));
    for r in results {
        println!(
            "  {:<8} {:>10} {:>12} {:>14.1}",
            r.tier, r.element_count, r.duration_us, r.throughput_mflops
        );
    }
}
