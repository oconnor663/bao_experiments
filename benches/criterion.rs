extern crate bao_experiments;
extern crate criterion;

use bao_experiments::*;
use criterion::*;
use std::mem;

fn bench_bao_standard(c: &mut Criterion) {
    c.bench(
        "throughput_benches",
        Benchmark::new("bench_bao_standard", |b| {
            let mut input = RandomInput::new(BENCH_LENGTH);
            b.iter(|| bao_standard(input.get()))
        })
        .throughput(Throughput::Bytes(BENCH_LENGTH as u32)),
    );
}

fn bench_bao_parallel_parents(c: &mut Criterion) {
    c.bench(
        "throughput_benches",
        Benchmark::new("bench_bao_parallel_parents", |b| {
            let mut input = RandomInput::new(BENCH_LENGTH);
            b.iter(|| bao_parallel_parents(input.get()))
        })
        .throughput(Throughput::Bytes(BENCH_LENGTH as u32)),
    );
}

fn bench_bao_large_chunks(c: &mut Criterion) {
    c.bench(
        "throughput_benches",
        Benchmark::new("bench_bao_large_chunks", |b| {
            let mut input = RandomInput::new(BENCH_LENGTH);
            b.iter(|| bao_large_chunks(input.get()))
        })
        .throughput(Throughput::Bytes(BENCH_LENGTH as u32)),
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_load_8_blake2s_blocks_simple(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_load_8_blake2s_blocks_interleave(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_load_8_blake2s_blocks_gather(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_load_8_blake2s_blocks_gather_inner(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_load_4_blake2b_blocks_simple(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_load_4_blake2b_blocks_interleave(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_load_4_blake2b_blocks_gather(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_load_4_blake2b_blocks_gather_inner(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
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
        bench_bao_standard,
        bench_bao_parallel_parents,
        bench_bao_large_chunks,
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
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

// The loading benches are only defined on x86.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
criterion_main!(throughput_benches, loading_benches);
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
criterion_main!(throughput_benches);
