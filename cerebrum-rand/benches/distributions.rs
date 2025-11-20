// ============================================================================
// CEREBRUM-RAND BENCHMARKS - Production Performance Analysis
// ============================================================================

use cerebrum_rand::*;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use half::{bf16, f16};
use rand::thread_rng;

// ============================================================================
// UNIFORM DISTRIBUTION BENCHMARKS
// ============================================================================

fn bench_uniform_f16_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("uniform_f16_single");
    let mut rng = thread_rng();
    let dist = Uniform::new(f16::from_f32(-1.0), f16::from_f32(1.0));

    group.bench_function("sample", |b| {
        b.iter(|| {
            let value = dist.sample(&mut rng);
            black_box(value)
        })
    });

    group.finish();
}

fn bench_uniform_f16_fill(c: &mut Criterion) {
    let mut group = c.benchmark_group("uniform_f16_fill");
    let mut rng = thread_rng();
    let dist = Uniform::new(f16::from_f32(-1.0), f16::from_f32(1.0));

    for size in [64, 256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut buffer = vec![f16::ZERO; size];
            b.iter(|| {
                dist.fill(&mut rng, &mut buffer);
                black_box(buffer[0])
            })
        });
    }

    group.finish();
}

fn bench_uniform_bf16_fill(c: &mut Criterion) {
    let mut group = c.benchmark_group("uniform_bf16_fill");
    let mut rng = thread_rng();
    let dist = Uniform::new(bf16::from_f32(-1.0), bf16::from_f32(1.0));

    for size in [64, 256, 1024, 4096].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut buffer = vec![bf16::ZERO; size];
            b.iter(|| {
                dist.fill(&mut rng, &mut buffer);
                black_box(buffer[0])
            })
        });
    }

    group.finish();
}

// ============================================================================
// NORMAL DISTRIBUTION BENCHMARKS
// ============================================================================

fn bench_normal_f16_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("normal_f16_single");
    let mut rng = thread_rng();
    let dist = Normal::new(f16::ZERO, f16::ONE);

    group.bench_function("sample", |b| {
        b.iter(|| {
            let value = dist.sample(&mut rng);
            black_box(value)
        })
    });

    group.finish();
}

fn bench_normal_f16_fill(c: &mut Criterion) {
    let mut group = c.benchmark_group("normal_f16_fill");
    let mut rng = thread_rng();
    let dist = Normal::new(f16::ZERO, f16::ONE);

    for size in [64, 256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut buffer = vec![f16::ZERO; size];
            b.iter(|| {
                dist.fill(&mut rng, &mut buffer);
                black_box(buffer[0])
            })
        });
    }

    group.finish();
}

fn bench_standard_normal_f16(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard_normal_f16");
    let mut rng = thread_rng();
    let dist = StandardNormal;

    group.bench_function("sample", |b| {
        b.iter(|| {
            let value: f16 = dist.sample(&mut rng);
            black_box(value)
        })
    });

    group.finish();
}

// ============================================================================
// XAVIER/GLOROT INITIALIZATION BENCHMARKS
// ============================================================================

fn bench_xavier_uniform_f16(c: &mut Criterion) {
    let mut group = c.benchmark_group("xavier_uniform_f16");
    let mut rng = thread_rng();

    // Typical layer sizes (fan_in x fan_out)
    let layer_configs = vec![
        (128, 64),   // Small layer
        (512, 256),  // Medium layer
        (2048, 1024), // Large layer
    ];

    for (fan_in, fan_out) in layer_configs {
        let dist = XavierUniform::new(fan_in, fan_out);
        let size = fan_in * fan_out;
        let id = BenchmarkId::new("weights", format!("{}x{}", fan_in, fan_out));

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(id, &size, |b, &size| {
            let mut weights = vec![f16::ZERO; size];
            b.iter(|| {
                dist.fill(&mut rng, &mut weights);
                black_box(weights[0])
            })
        });
    }

    group.finish();
}

fn bench_xavier_normal_f16(c: &mut Criterion) {
    let mut group = c.benchmark_group("xavier_normal_f16");
    let mut rng = thread_rng();

    let layer_configs = vec![(128, 64), (512, 256), (2048, 1024)];

    for (fan_in, fan_out) in layer_configs {
        let dist = XavierNormal::new(fan_in, fan_out);
        let size = fan_in * fan_out;
        let id = BenchmarkId::new("weights", format!("{}x{}", fan_in, fan_out));

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(id, &size, |b, &size| {
            let mut weights = vec![f16::ZERO; size];
            b.iter(|| {
                dist.fill(&mut rng, &mut weights);
                black_box(weights[0])
            })
        });
    }

    group.finish();
}

// ============================================================================
// KAIMING/HE INITIALIZATION BENCHMARKS
// ============================================================================

fn bench_kaiming_uniform_f16(c: &mut Criterion) {
    let mut group = c.benchmark_group("kaiming_uniform_f16");
    let mut rng = thread_rng();

    let layer_configs = vec![
        (128, 64),
        (512, 256),
        (2048, 1024),
    ];

    for (fan_in, fan_out) in layer_configs {
        let dist = KaimingUniform::new(fan_in);
        let size = fan_in * fan_out;
        let id = BenchmarkId::new("weights", format!("{}x{}", fan_in, fan_out));

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(id, &size, |b, &size| {
            let mut weights = vec![f16::ZERO; size];
            b.iter(|| {
                dist.fill(&mut rng, &mut weights);
                black_box(weights[0])
            })
        });
    }

    group.finish();
}

fn bench_kaiming_normal_f16(c: &mut Criterion) {
    let mut group = c.benchmark_group("kaiming_normal_f16");
    let mut rng = thread_rng();

    let layer_configs = vec![(128, 64), (512, 256), (2048, 1024)];

    for (fan_in, fan_out) in layer_configs {
        let dist = KaimingNormal::new(fan_in);
        let size = fan_in * fan_out;
        let id = BenchmarkId::new("weights", format!("{}x{}", fan_in, fan_out));

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(id, &size, |b, &size| {
            let mut weights = vec![f16::ZERO; size];
            b.iter(|| {
                dist.fill(&mut rng, &mut weights);
                black_box(weights[0])
            })
        });
    }

    group.finish();
}

// ============================================================================
// CANDLE-STYLE FREE FUNCTIONS BENCHMARKS
// ============================================================================

fn bench_free_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("free_functions");
    let mut rng = thread_rng();

    group.bench_function("standard_uniform_f16", |b| {
        b.iter(|| {
            let value = standard_uniform_f16(&mut rng);
            black_box(value)
        })
    });

    group.bench_function("standard_normal_f16", |b| {
        b.iter(|| {
            let value = standard_normal_f16(&mut rng);
            black_box(value)
        })
    });

    group.bench_function("uniform_f16", |b| {
        b.iter(|| {
            let value = uniform_f16(&mut rng, f16::from_f32(-1.0), f16::from_f32(1.0));
            black_box(value)
        })
    });

    group.bench_function("normal_f16", |b| {
        b.iter(|| {
            let value = normal_f16(&mut rng, f16::ZERO, f16::ONE);
            black_box(value)
        })
    });

    group.finish();
}

// ============================================================================
// SEEDED RNG BENCHMARKS
// ============================================================================

fn bench_seeded_rng(c: &mut Criterion) {
    let mut group = c.benchmark_group("seeded_rng");
    let dist = Normal::new(f16::ZERO, f16::ONE);

    group.bench_function("reproducible_generation", |b| {
        b.iter(|| {
            let mut rng = seeded_rng(42);
            let mut buffer = vec![f16::ZERO; 1024];
            dist.fill(&mut rng, &mut buffer);
            black_box(buffer[0])
        })
    });

    group.finish();
}

// ============================================================================
// COMPARISON: f16 vs bf16
// ============================================================================

fn bench_f16_vs_bf16(c: &mut Criterion) {
    let mut group = c.benchmark_group("f16_vs_bf16_comparison");
    let mut rng = thread_rng();

    let size = 4096;

    // f16 uniform
    group.bench_function("uniform_f16", |b| {
        let dist = Uniform::new(f16::from_f32(-1.0), f16::from_f32(1.0));
        let mut buffer = vec![f16::ZERO; size];
        b.iter(|| {
            dist.fill(&mut rng, &mut buffer);
            black_box(buffer[0])
        })
    });

    // bf16 uniform
    group.bench_function("uniform_bf16", |b| {
        let dist = Uniform::new(bf16::from_f32(-1.0), bf16::from_f32(1.0));
        let mut buffer = vec![bf16::ZERO; size];
        b.iter(|| {
            dist.fill(&mut rng, &mut buffer);
            black_box(buffer[0])
        })
    });

    // f16 normal
    group.bench_function("normal_f16", |b| {
        let dist = Normal::new(f16::ZERO, f16::ONE);
        let mut buffer = vec![f16::ZERO; size];
        b.iter(|| {
            dist.fill(&mut rng, &mut buffer);
            black_box(buffer[0])
        })
    });

    // bf16 normal
    group.bench_function("normal_bf16", |b| {
        let dist = Normal::new(bf16::ZERO, bf16::ONE);
        let mut buffer = vec![bf16::ZERO; size];
        b.iter(|| {
            dist.fill(&mut rng, &mut buffer);
            black_box(buffer[0])
        })
    });

    group.finish();
}

// ============================================================================
// BENCHMARK GROUP REGISTRATION
// ============================================================================

criterion_group!(
    benches,
    // Uniform distributions
    bench_uniform_f16_single,
    bench_uniform_f16_fill,
    bench_uniform_bf16_fill,
    // Normal distributions
    bench_normal_f16_single,
    bench_normal_f16_fill,
    bench_standard_normal_f16,
    // Xavier/Glorot
    bench_xavier_uniform_f16,
    bench_xavier_normal_f16,
    // Kaiming/He
    bench_kaiming_uniform_f16,
    bench_kaiming_normal_f16,
    // Free functions
    bench_free_functions,
    // Seeded RNG
    bench_seeded_rng,
    // Comparisons
    bench_f16_vs_bf16,
);

criterion_main!(benches);
