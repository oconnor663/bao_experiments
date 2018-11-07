extern crate bao_experiments;
extern crate criterion;

use bao_experiments::Finalization::*;
use criterion::*;
use std::time::Duration;

// The current 4ary implementation is only defined for inputs that are a power
// of 4 times the chunk size. 2^24 bytes is about 17 MB.
const LENGTH: usize = 1 << 24;
// A long warmup seems important for getting consistent numbers. Intel "Turbo Boost" makes a big
// difference, but after a few seconds the processor heats up and turns it off.
const WARMUP_SECS: u64 = 10;

fn bench_blake2s(c: &mut Criterion) {
    c.bench(
        "throughput_benches",
        Benchmark::new("bench_blake2s", |b| {
            let input = vec![0xff; LENGTH];
            b.iter(move || bao_experiments::hash_recurse_rayon_blake2s(&input, Root(LENGTH as u64)))
        }).throughput(Throughput::Bytes(LENGTH as u32)),
    );
}

fn bench_blake2b_standard(c: &mut Criterion) {
    c.bench(
        "throughput_benches",
        Benchmark::new("bench_blake2b_standard", |b| {
            let input = vec![0xff; LENGTH];
            b.iter(move || bao_experiments::hash_recurse_rayon_blake2b(&input, Root(LENGTH as u64)))
        }).throughput(Throughput::Bytes(LENGTH as u32)),
    );
}

fn bench_blake2b_standard_parallel_parents(c: &mut Criterion) {
    c.bench(
        "throughput_benches",
        Benchmark::new("bench_blake2b_standard_parallel_parents", |b| {
            let input = vec![0xff; LENGTH];
            b.iter(move || bao_experiments::hash_recurse_rayon_blake2b_parallel_parents(&input))
        }).throughput(Throughput::Bytes(LENGTH as u32)),
    );
}

fn bench_blake2b_4ary(c: &mut Criterion) {
    c.bench(
        "throughput_benches",
        Benchmark::new("bench_blake2b_4ary", |b| {
            let input = vec![0xff; LENGTH];
            b.iter(move || {
                bao_experiments::hash_recurse_rayon_blake2b_4ary(&input, Root(LENGTH as u64))
            })
        }).throughput(Throughput::Bytes(LENGTH as u32)),
    );
}

fn bench_blake2b_4ary_parallel_parents(c: &mut Criterion) {
    c.bench(
        "throughput_benches",
        Benchmark::new("bench_blake2b_4ary_parallel_parents", |b| {
            let input = vec![0xff; LENGTH];
            b.iter(move || {
                bao_experiments::hash_recurse_rayon_blake2b_4ary_parallel_parents(&input)
            })
        }).throughput(Throughput::Bytes(LENGTH as u32)),
    );
}

fn bench_blake2b_large_chunks(c: &mut Criterion) {
    c.bench(
        "throughput_benches",
        Benchmark::new("bench_blake2b_large_chunks", |b| {
            let input = vec![0xff; LENGTH];
            b.iter(move || {
                bao_experiments::hash_recurse_rayon_blake2b_large_chunks(
                    &input,
                    Root(LENGTH as u64),
                )
            })
        }).throughput(Throughput::Bytes(LENGTH as u32)),
    );
}

// NOTE: This benchmark is slower than it should be, for lack of an SSE implementation of BLAKE2s.
fn bench_blake2hybrid(c: &mut Criterion) {
    c.bench(
        "throughput_benches",
        Benchmark::new("bench_blake2hybrid", |b| {
            let input = vec![0xff; LENGTH];
            b.iter(move || {
                bao_experiments::hash_recurse_rayon_blake2hybrid(&input, Root(LENGTH as u64))
            })
        }).throughput(Throughput::Bytes(LENGTH as u32)),
    );
}

fn bench_blake2hybrid_parallel_parents(c: &mut Criterion) {
    c.bench(
        "throughput_benches",
        Benchmark::new("bench_blake2hybrid_parallel_parents", |b| {
            let input = vec![0xff; LENGTH];
            b.iter(move || {
                bao_experiments::hash_recurse_rayon_blake2hybrid_parallel_parents(
                    &input,
                    Root(LENGTH as u64),
                )
            })
        }).throughput(Throughput::Bytes(LENGTH as u32)),
    );
}

criterion_group!(
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_secs(WARMUP_SECS));
    targets =
        bench_blake2s,
        bench_blake2b_standard,
        bench_blake2b_standard_parallel_parents,
        bench_blake2b_4ary,
        bench_blake2b_4ary_parallel_parents,
        bench_blake2b_large_chunks,
        bench_blake2hybrid,
        bench_blake2hybrid_parallel_parents,
);
criterion_main!(benches);
