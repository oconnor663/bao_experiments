#![feature(test)]

extern crate bao_experiments;
extern crate test;

use bao_experiments::*;
use test::Bencher;

// The 4ary implementation is only defined for inputs that are a power of 4 times the chunk size.
const LENGTH: usize = 1 << 24; // about 17 MB

fn input(b: &mut Bencher, size: usize) -> Vec<u8> {
    b.bytes = size as u64;
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
    let block0 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    let block1 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    let block2 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    let block3 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    let block4 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    let block5 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    let block6 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    let block7 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    b.iter(|| unsafe {
        avx2_blake2s_load::load_msg_vecs_simple(
            &block0, &block1, &block2, &block3, &block4, &block5, &block6, &block7,
        )
    })
}

#[bench]
fn bench_load_8_blake2s_blocks_interleave(b: &mut Bencher) {
    let block0 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    let block1 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    let block2 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    let block3 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    let block4 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    let block5 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    let block6 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    let block7 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    b.iter(|| unsafe {
        avx2_blake2s_load::load_msg_vecs_interleave(
            &block0, &block1, &block2, &block3, &block4, &block5, &block6, &block7,
        )
    })
}

#[bench]
fn bench_load_8_blake2s_blocks_gather(b: &mut Bencher) {
    let block0 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    let block1 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    let block2 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    let block3 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    let block4 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    let block5 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    let block6 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    let block7 = [0; avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    b.iter(|| unsafe {
        avx2_blake2s_load::load_msg_vecs_gather(
            &block0, &block1, &block2, &block3, &block4, &block5, &block6, &block7,
        )
    })
}

#[bench]
fn bench_load_8_blake2s_blocks_gather_inner(b: &mut Bencher) {
    let blocks = [1; 8 * avx2_blake2s_load::BLAKE2S_BLOCKBYTES];
    b.iter(|| unsafe { avx2_blake2s_load::gather_from_blocks(&blocks) })
}

#[bench]
fn bench_load_8_blake2b_blocks_simple(b: &mut Bencher) {
    let block0 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
    let block1 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
    let block2 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
    let block3 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
    b.iter(|| unsafe {
        avx2_blake2b_load::load_msg_vecs_simple(&block0, &block1, &block2, &block3)
    })
}

#[bench]
fn bench_load_8_blake2b_blocks_interleave(b: &mut Bencher) {
    let block0 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
    let block1 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
    let block2 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
    let block3 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
    b.iter(|| unsafe {
        avx2_blake2b_load::load_msg_vecs_interleave(&block0, &block1, &block2, &block3)
    })
}

#[bench]
fn bench_load_8_blake2b_blocks_gather(b: &mut Bencher) {
    let block0 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
    let block1 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
    let block2 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
    let block3 = [0; avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
    b.iter(|| unsafe {
        avx2_blake2b_load::load_msg_vecs_gather(&block0, &block1, &block2, &block3)
    })
}

#[bench]
fn bench_load_8_blake2b_blocks_gather_inner(b: &mut Bencher) {
    let blocks = [1; 4 * avx2_blake2b_load::BLAKE2B_BLOCKBYTES];
    b.iter(|| unsafe { avx2_blake2b_load::gather_from_blocks(&blocks) })
}
