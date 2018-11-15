#![feature(test)]

extern crate bao_experiments;
extern crate rand;
extern crate test;

use bao_experiments::*;
use std::mem;
use test::Bencher;

// The 4ary implementation is only defined for inputs that are a power of 4 times the chunk size.
const LENGTH: usize = 1 << 24; // about 17 MB

fn input(b: &mut Bencher, size: usize) -> Vec<u8> {
    b.bytes = size as u64;
    // TRICKY BENCHMARKING DETAIL! It's important to avoid using all-zero memory as input:
    // - The allocator might return uninitialized pages, which get zeroed lazily when they're read.
    //   In that case, the first iteration pays the cost of initializnig the memory, which makes
    //   your throughput lower and less consistent.
    // - For some reason I don't understand, benchmarks on a giant 48-physical-core AWS machine are
    //   20% *faster* when the input is all-zeros. There might be some other effect that comes into
    //   play with the gigantic inputs we use in those benchmarks.
    vec![0xff; size]
}

#[bench]
fn bench_blake2b_standard(b: &mut Bencher) {
    let input = input(b, LENGTH);
    b.iter(|| bao_standard(&input));
}

#[bench]
fn bench_blake2b_standard_parallel_parents(b: &mut Bencher) {
    let input = input(b, LENGTH);
    b.iter(|| bao_standard_parallel_parents(&input));
}

#[bench]
fn bench_blake2s(b: &mut Bencher) {
    let input = input(b, LENGTH);
    b.iter(|| bao_blake2s(&input));
}

#[bench]
fn bench_blake2s_parallel_parents(b: &mut Bencher) {
    let input = input(b, LENGTH);
    b.iter(|| bao_blake2s_parallel_parents(&input));
}

#[bench]
fn bench_blake2b_4ary(b: &mut Bencher) {
    let input = input(b, LENGTH);
    b.iter(|| bao_4ary(&input));
}

#[bench]
fn bench_blake2b_4ary_parallel_parents(b: &mut Bencher) {
    let input = input(b, LENGTH);
    b.iter(|| bao_4ary_parallel_parents(&input));
}

#[bench]
fn bench_blake2b_large_chunks(b: &mut Bencher) {
    let input = input(b, LENGTH);
    b.iter(|| bao_blake2b_large_chunks(&input));
}

#[bench]
fn bench_blake2s_large_chunks(b: &mut Bencher) {
    let input = input(b, LENGTH);
    b.iter(|| bao_blake2s_large_chunks(&input));
}

// NOTE: This benchmark is slower than it should be, for lack of an SSE implementation of BLAKE2s.
#[bench]
fn bench_blake2hybrid(b: &mut Bencher) {
    let input = input(b, LENGTH);
    b.iter(|| bao_blake2hybrid(&input));
}

#[bench]
fn bench_blake2hybrid_parallel_parents(b: &mut Bencher) {
    let input = input(b, LENGTH);
    b.iter(|| bao_blake2hybrid_parallel_parents(&input));
}

#[bench]
fn bench_load_8_blake2s_blocks_simple(b: &mut Bencher) {
    let block0 = avx2_blake2s_load::random_block();
    let block1 = avx2_blake2s_load::random_block();
    let block2 = avx2_blake2s_load::random_block();
    let block3 = avx2_blake2s_load::random_block();
    let block4 = avx2_blake2s_load::random_block();
    let block5 = avx2_blake2s_load::random_block();
    let block6 = avx2_blake2s_load::random_block();
    let block7 = avx2_blake2s_load::random_block();
    let mut out = unsafe { mem::zeroed() };
    b.iter(|| unsafe {
        avx2_blake2s_load::load_msg_vecs_simple(
            &block0, &block1, &block2, &block3, &block4, &block5, &block6, &block7, &mut out,
        );
        test::black_box(&mut out);
    })
}

#[bench]
fn bench_load_8_blake2s_blocks_interleave(b: &mut Bencher) {
    let block0 = avx2_blake2s_load::random_block();
    let block1 = avx2_blake2s_load::random_block();
    let block2 = avx2_blake2s_load::random_block();
    let block3 = avx2_blake2s_load::random_block();
    let block4 = avx2_blake2s_load::random_block();
    let block5 = avx2_blake2s_load::random_block();
    let block6 = avx2_blake2s_load::random_block();
    let block7 = avx2_blake2s_load::random_block();
    let mut out = unsafe { mem::zeroed() };
    b.iter(|| unsafe {
        avx2_blake2s_load::load_msg_vecs_interleave(
            &block0, &block1, &block2, &block3, &block4, &block5, &block6, &block7, &mut out,
        );
        test::black_box(&mut out);
    })
}

#[bench]
fn bench_load_8_blake2s_blocks_gather(b: &mut Bencher) {
    let block0 = avx2_blake2s_load::random_block();
    let block1 = avx2_blake2s_load::random_block();
    let block2 = avx2_blake2s_load::random_block();
    let block3 = avx2_blake2s_load::random_block();
    let block4 = avx2_blake2s_load::random_block();
    let block5 = avx2_blake2s_load::random_block();
    let block6 = avx2_blake2s_load::random_block();
    let block7 = avx2_blake2s_load::random_block();
    let mut out = unsafe { mem::uninitialized() };
    b.iter(|| unsafe {
        avx2_blake2s_load::load_msg_vecs_gather(
            &block0, &block1, &block2, &block3, &block4, &block5, &block6, &block7, &mut out,
        );
        test::black_box(&mut out);
    })
}

#[bench]
fn bench_load_8_blake2s_blocks_gather_inner(b: &mut Bencher) {
    let blocks = avx2_blake2s_load::random_8_blocks();
    let mut out = unsafe { mem::zeroed() };
    b.iter(|| unsafe {
        avx2_blake2s_load::gather_from_blocks(&blocks, &mut out);
        test::black_box(&mut out);
    })
}

#[bench]
fn bench_load_4_blake2b_blocks_simple(b: &mut Bencher) {
    let block0 = avx2_blake2b_load::random_block();
    let block1 = avx2_blake2b_load::random_block();
    let block2 = avx2_blake2b_load::random_block();
    let block3 = avx2_blake2b_load::random_block();
    let mut out = unsafe { mem::zeroed() };
    b.iter(|| unsafe {
        avx2_blake2b_load::load_msg_vecs_simple(&block0, &block1, &block2, &block3, &mut out);
        test::black_box(&mut out);
    })
}

#[bench]
fn bench_load_4_blake2b_blocks_interleave(b: &mut Bencher) {
    let block0 = avx2_blake2b_load::random_block();
    let block1 = avx2_blake2b_load::random_block();
    let block2 = avx2_blake2b_load::random_block();
    let block3 = avx2_blake2b_load::random_block();
    let mut out = unsafe { mem::zeroed() };
    b.iter(|| unsafe {
        avx2_blake2b_load::load_msg_vecs_interleave(&block0, &block1, &block2, &block3, &mut out);
        test::black_box(&mut out);
    })
}

#[bench]
fn bench_load_4_blake2b_blocks_gather(b: &mut Bencher) {
    let block0 = avx2_blake2b_load::random_block();
    let block1 = avx2_blake2b_load::random_block();
    let block2 = avx2_blake2b_load::random_block();
    let block3 = avx2_blake2b_load::random_block();
    let mut out = unsafe { mem::zeroed() };
    b.iter(|| unsafe {
        avx2_blake2b_load::load_msg_vecs_gather(&block0, &block1, &block2, &block3, &mut out);
        test::black_box(&mut out);
    })
}

#[bench]
fn bench_load_4_blake2b_blocks_gather_inner(b: &mut Bencher) {
    let blocks = avx2_blake2b_load::random_4_blocks();
    let mut out = unsafe { mem::zeroed() };
    b.iter(|| unsafe {
        avx2_blake2b_load::gather_from_blocks(&blocks, &mut out);
        test::black_box(&mut out);
    })
}
