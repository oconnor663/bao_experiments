extern crate bao_experiments;
extern crate criterion;

use bao_experiments::*;
use criterion::*;
use std::time::Duration;

// The current 4ary implementation is only defined for inputs that are a power
// of 4 times the chunk size. 2^24 bytes is about 17 MB.
const LENGTH: usize = 1 << 24;
// A long warmup seems important for getting consistent numbers. Intel "Turbo Boost" makes a big
// difference, but after a few seconds the processor heats up and turns it off.
const WARMUP_SECS: u64 = 10;

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
            let block0 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            let block1 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            let block2 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            let block3 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            let block4 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            let block5 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            let block6 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            let block7 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            b.iter(move || unsafe {
                avx2_blake2s_load::load_msg_vecs_simple(
                    &block0, &block1, &block2, &block3, &block4, &block5, &block6, &block7,
                )
            })
        }),
    );
}
fn bench_load_8_blake2s_blocks_interleave(c: &mut Criterion) {
    c.bench(
        "loading_benches",
        Benchmark::new("bench_load_8_blake2s_blocks_interleave", |b| {
            let block0 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            let block1 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            let block2 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            let block3 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            let block4 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            let block5 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            let block6 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            let block7 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            b.iter(move || unsafe {
                avx2_blake2s_load::load_msg_vecs_interleave(
                    &block0, &block1, &block2, &block3, &block4, &block5, &block6, &block7,
                )
            })
        }),
    );
}
fn bench_load_8_blake2s_blocks_gather(c: &mut Criterion) {
    c.bench(
        "loading_benches",
        Benchmark::new("bench_load_8_blake2s_blocks_gather", |b| {
            let block0 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            let block1 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            let block2 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            let block3 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            let block4 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            let block5 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            let block6 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            let block7 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            b.iter(move || unsafe {
                avx2_blake2s_load::load_msg_vecs_gather(
                    &block0, &block1, &block2, &block3, &block4, &block5, &block6, &block7,
                )
            })
        }),
    );
}
fn bench_load_8_blake2s_blocks_gather_inner(c: &mut Criterion) {
    c.bench(
        "loading_benches",
        Benchmark::new("bench_load_8_blake2s_blocks_gather_inner", |b| {
            let blocks = [1; 8 * avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
            b.iter(move || unsafe { avx2_blake2s_load::gather_from_blocks(&blocks) })
        }),
    );
}

fn bench_load_4_blake2b_blocks_simple(c: &mut Criterion) {
    c.bench(
        "loading_benches",
        Benchmark::new("bench_load_4_blake2b_blocks_simple", |b| {
            let block0 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
            let block1 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
            let block2 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
            let block3 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
            b.iter(move || unsafe {
                avx2_blake2b_load::load_msg_vecs_simple(&block0, &block1, &block2, &block3)
            })
        }),
    );
}
fn bench_load_4_blake2b_blocks_interleave(c: &mut Criterion) {
    c.bench(
        "loading_benches",
        Benchmark::new("bench_load_4_blake2b_blocks_interleave", |b| {
            let block0 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
            let block1 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
            let block2 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
            let block3 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
            b.iter(move || unsafe {
                avx2_blake2b_load::load_msg_vecs_interleave(&block0, &block1, &block2, &block3)
            })
        }),
    );
}
fn bench_load_4_blake2b_blocks_gather(c: &mut Criterion) {
    c.bench(
        "loading_benches",
        Benchmark::new("bench_load_4_blake2b_blocks_gather", |b| {
            let block0 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
            let block1 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
            let block2 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
            let block3 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
            b.iter(move || unsafe {
                avx2_blake2b_load::load_msg_vecs_gather(&block0, &block1, &block2, &block3)
            })
        }),
    );
}
fn bench_load_4_blake2b_blocks_gather_inner(c: &mut Criterion) {
    c.bench(
        "loading_benches",
        Benchmark::new("bench_load_4_blake2b_blocks_gather_inner", |b| {
            let blocks = [1; 4 * avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
            b.iter(move || unsafe { avx2_blake2b_load::gather_from_blocks(&blocks) })
        }),
    );
}

criterion_group!(
    name = throughput_benches;
    config = Criterion::default().warm_up_time(Duration::from_secs(WARMUP_SECS));
    targets =
        bench_blake2b_standard,
        bench_blake2b_standard_parallel_parents,
        bench_blake2s,
        bench_blake2s_parallel_parents,
        bench_blake2b_4ary,
        bench_blake2b_4ary_parallel_parents,
        bench_blake2b_large_chunks,
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
