extern crate bao_experiments;
extern crate criterion;

use bao_experiments::*;
use criterion::*;
use std::mem;

// The current 4ary implementation is only defined for inputs that are a power
// of 4 times the chunk size. 2^24 bytes is about 17 MB.
const LENGTH: usize = 1 << 24;

fn bench_blake2b_standard(c: &mut Criterion) {
    c.bench(
        "throughput_benches",
        Benchmark::new("bench_blake2b_standard", |b| {
            let input = vec![0xff; LENGTH];
            b.iter(move || bao_standard(&input))
        }).throughput(Throughput::Bytes(LENGTH as u32)),
    );
}

fn bench_blake2b_standard_parallel_parents(c: &mut Criterion) {
    c.bench(
        "throughput_benches",
        Benchmark::new("bench_blake2b_standard_parallel_parents", |b| {
            let input = vec![0xff; LENGTH];
            b.iter(move || bao_standard_parallel_parents(&input))
        }).throughput(Throughput::Bytes(LENGTH as u32)),
    );
}

fn bench_blake2s(c: &mut Criterion) {
    c.bench(
        "throughput_benches",
        Benchmark::new("bench_blake2s", |b| {
            let input = vec![0xff; LENGTH];
            b.iter(move || bao_blake2s(&input))
        }).throughput(Throughput::Bytes(LENGTH as u32)),
    );
}

fn bench_blake2s_parallel_parents(c: &mut Criterion) {
    c.bench(
        "throughput_benches",
        Benchmark::new("bench_blake2s_parallel_parents", |b| {
            let input = vec![0xff; LENGTH];
            b.iter(move || bao_blake2s_parallel_parents(&input))
        }).throughput(Throughput::Bytes(LENGTH as u32)),
    );
}

fn bench_blake2b_4ary(c: &mut Criterion) {
    c.bench(
        "throughput_benches",
        Benchmark::new("bench_blake2b_4ary", |b| {
            let input = vec![0xff; LENGTH];
            b.iter(move || bao_4ary(&input))
        }).throughput(Throughput::Bytes(LENGTH as u32)),
    );
}

fn bench_blake2b_4ary_parallel_parents(c: &mut Criterion) {
    c.bench(
        "throughput_benches",
        Benchmark::new("bench_blake2b_4ary_parallel_parents", |b| {
            let input = vec![0xff; LENGTH];
            b.iter(move || bao_4ary_parallel_parents(&input))
        }).throughput(Throughput::Bytes(LENGTH as u32)),
    );
}

fn bench_blake2b_large_chunks(c: &mut Criterion) {
    c.bench(
        "throughput_benches",
        Benchmark::new("bench_blake2b_large_chunks", |b| {
            let input = vec![0xff; LENGTH];
            b.iter(move || bao_blake2b_large_chunks(&input))
        }).throughput(Throughput::Bytes(LENGTH as u32)),
    );
}

fn bench_blake2s_large_chunks(c: &mut Criterion) {
    c.bench(
        "throughput_benches",
        Benchmark::new("bench_blake2s_large_chunks", |b| {
            let input = vec![0xff; LENGTH];
            b.iter(move || bao_blake2s_large_chunks(&input))
        }).throughput(Throughput::Bytes(LENGTH as u32)),
    );
}

// NOTE: This benchmark is slower than it should be, for lack of an SSE implementation of BLAKE2s.
fn bench_blake2hybrid(c: &mut Criterion) {
    c.bench(
        "throughput_benches",
        Benchmark::new("bench_blake2hybrid", |b| {
            let input = vec![0xff; LENGTH];
            b.iter(move || bao_blake2hybrid(&input))
        }).throughput(Throughput::Bytes(LENGTH as u32)),
    );
}

fn bench_blake2hybrid_parallel_parents(c: &mut Criterion) {
    c.bench(
        "throughput_benches",
        Benchmark::new("bench_blake2hybrid_parallel_parents", |b| {
            let input = vec![0xff; LENGTH];
            b.iter(move || bao_blake2hybrid_parallel_parents(&input))
        }).throughput(Throughput::Bytes(LENGTH as u32)),
    );
}

fn bench_load_8_blake2s_blocks_simple(c: &mut Criterion) {
    c.bench(
        "loading_benches",
        Benchmark::new("bench_load_8_blake2s_blocks_simple", |b| {
            let block0 = avx2_blake2s_load::random_block();
            let block1 = avx2_blake2s_load::random_block();
            let block2 = avx2_blake2s_load::random_block();
            let block3 = avx2_blake2s_load::random_block();
            let block4 = avx2_blake2s_load::random_block();
            let block5 = avx2_blake2s_load::random_block();
            let block6 = avx2_blake2s_load::random_block();
            let block7 = avx2_blake2s_load::random_block();
            let mut out = unsafe { mem::zeroed() };
            b.iter(move || unsafe {
                avx2_blake2s_load::load_msg_vecs_simple(
                    &block0, &block1, &block2, &block3, &block4, &block5, &block6, &block7,
                    &mut out,
                );
                criterion::black_box(&out);
            });
        }),
    );
}
fn bench_load_8_blake2s_blocks_interleave(c: &mut Criterion) {
    c.bench(
        "loading_benches",
        Benchmark::new("bench_load_8_blake2s_blocks_interleave", |b| {
            let block0 = avx2_blake2s_load::random_block();
            let block1 = avx2_blake2s_load::random_block();
            let block2 = avx2_blake2s_load::random_block();
            let block3 = avx2_blake2s_load::random_block();
            let block4 = avx2_blake2s_load::random_block();
            let block5 = avx2_blake2s_load::random_block();
            let block6 = avx2_blake2s_load::random_block();
            let block7 = avx2_blake2s_load::random_block();
            let mut out = unsafe { mem::zeroed() };
            b.iter(move || unsafe {
                avx2_blake2s_load::load_msg_vecs_interleave(
                    &block0, &block1, &block2, &block3, &block4, &block5, &block6, &block7,
                    &mut out,
                );
                criterion::black_box(&out);
            });
        }),
    );
}
fn bench_load_8_blake2s_blocks_gather(c: &mut Criterion) {
    c.bench(
        "loading_benches",
        Benchmark::new("bench_load_8_blake2s_blocks_gather", |b| {
            let block0 = avx2_blake2s_load::random_block();
            let block1 = avx2_blake2s_load::random_block();
            let block2 = avx2_blake2s_load::random_block();
            let block3 = avx2_blake2s_load::random_block();
            let block4 = avx2_blake2s_load::random_block();
            let block5 = avx2_blake2s_load::random_block();
            let block6 = avx2_blake2s_load::random_block();
            let block7 = avx2_blake2s_load::random_block();
            let mut out = unsafe { mem::zeroed() };
            b.iter(move || unsafe {
                avx2_blake2s_load::load_msg_vecs_gather(
                    &block0, &block1, &block2, &block3, &block4, &block5, &block6, &block7,
                    &mut out,
                );
                criterion::black_box(&out);
            });
        }),
    );
}
fn bench_load_8_blake2s_blocks_gather_inner(c: &mut Criterion) {
    c.bench(
        "loading_benches",
        Benchmark::new("bench_load_8_blake2s_blocks_gather_inner", |b| {
            let blocks = avx2_blake2s_load::random_8_blocks();
            let mut out = unsafe { mem::zeroed() };
            b.iter(move || unsafe {
                avx2_blake2s_load::gather_from_blocks(&blocks, &mut out);
                criterion::black_box(&mut out);
            });
        }),
    );
}

fn bench_load_4_blake2b_blocks_simple(c: &mut Criterion) {
    c.bench(
        "loading_benches",
        Benchmark::new("bench_load_4_blake2b_blocks_simple", |b| {
            let block0 = avx2_blake2b_load::random_block();
            let block1 = avx2_blake2b_load::random_block();
            let block2 = avx2_blake2b_load::random_block();
            let block3 = avx2_blake2b_load::random_block();
            let mut out = unsafe { mem::zeroed() };
            b.iter(move || unsafe {
                avx2_blake2b_load::load_msg_vecs_simple(
                    &block0, &block1, &block2, &block3, &mut out,
                );
                criterion::black_box(&mut out);
            });
        }),
    );
}
fn bench_load_4_blake2b_blocks_interleave(c: &mut Criterion) {
    c.bench(
        "loading_benches",
        Benchmark::new("bench_load_4_blake2b_blocks_interleave", |b| {
            let block0 = avx2_blake2b_load::random_block();
            let block1 = avx2_blake2b_load::random_block();
            let block2 = avx2_blake2b_load::random_block();
            let block3 = avx2_blake2b_load::random_block();
            let mut out = unsafe { mem::zeroed() };
            b.iter(move || unsafe {
                avx2_blake2b_load::load_msg_vecs_interleave(
                    &block0, &block1, &block2, &block3, &mut out,
                );
                criterion::black_box(&mut out);
            });
        }),
    );
}
fn bench_load_4_blake2b_blocks_gather(c: &mut Criterion) {
    c.bench(
        "loading_benches",
        Benchmark::new("bench_load_4_blake2b_blocks_gather", |b| {
            let block0 = avx2_blake2b_load::random_block();
            let block1 = avx2_blake2b_load::random_block();
            let block2 = avx2_blake2b_load::random_block();
            let block3 = avx2_blake2b_load::random_block();
            let mut out = unsafe { mem::zeroed() };
            b.iter(move || unsafe {
                avx2_blake2b_load::load_msg_vecs_gather(
                    &block0, &block1, &block2, &block3, &mut out,
                );
                criterion::black_box(&mut out);
            });
        }),
    );
}
fn bench_load_4_blake2b_blocks_gather_inner(c: &mut Criterion) {
    c.bench(
        "loading_benches",
        Benchmark::new("bench_load_4_blake2b_blocks_gather_inner", |b| {
            let blocks = avx2_blake2b_load::random_4_blocks();
            let mut out = unsafe { mem::zeroed() };
            b.iter(move || unsafe {
                avx2_blake2b_load::gather_from_blocks(&blocks, &mut out);
                criterion::black_box(&mut out);
            });
        }),
    );
}

criterion_group!(
    name = throughput_benches;
    config = Criterion::default();
    targets =
        bench_blake2b_standard,
        bench_blake2b_standard_parallel_parents,
        bench_blake2s,
        bench_blake2s_parallel_parents,
        bench_blake2b_4ary,
        bench_blake2b_4ary_parallel_parents,
        bench_blake2b_large_chunks,
        bench_blake2s_large_chunks,
        bench_blake2hybrid,
        bench_blake2hybrid_parallel_parents,
);
criterion_group!(
    name = loading_benches;
    config = Criterion::default();
    targets =
        bench_load_8_blake2s_blocks_simple,
        bench_load_8_blake2s_blocks_interleave,
        bench_load_8_blake2s_blocks_gather,
        bench_load_8_blake2s_blocks_gather_inner,
        bench_load_4_blake2b_blocks_simple,
        bench_load_4_blake2b_blocks_interleave,
        bench_load_4_blake2b_blocks_gather,
        bench_load_4_blake2b_blocks_gather_inner,
);
criterion_main!(throughput_benches, loading_benches);
