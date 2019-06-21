#![feature(test)]

extern crate bao_experiments;
extern crate rand;
extern crate test;

use bao_experiments::*;
use std::mem;
use test::Bencher;

#[bench]
fn bench_bao_basic_1_small(b: &mut Bencher) {
    b.bytes = BENCH_LENGTH as u64;
    let mut input = RandomInput::new(BENCH_LENGTH);
    b.iter(|| bao_basic(input.get(), SMALL_CHUNK_SIZE));
}

#[bench]
fn bench_bao_basic_2_medium(b: &mut Bencher) {
    b.bytes = BENCH_LENGTH as u64;
    let mut input = RandomInput::new(BENCH_LENGTH);
    b.iter(|| bao_basic(input.get(), CHUNK_SIZE));
}

#[bench]
fn bench_bao_basic_3_large(b: &mut Bencher) {
    b.bytes = BENCH_LENGTH as u64;
    let mut input = RandomInput::new(BENCH_LENGTH);
    b.iter(|| bao_basic(input.get(), LARGE_CHUNK_SIZE));
}

#[bench]
fn bench_bao_parallel_parents_1_small(b: &mut Bencher) {
    b.bytes = BENCH_LENGTH as u64;
    let mut input = RandomInput::new(BENCH_LENGTH);
    b.iter(|| bao_parallel_parents(input.get(), SMALL_CHUNK_SIZE));
}

#[bench]
fn bench_bao_parallel_parents_2_medium(b: &mut Bencher) {
    b.bytes = BENCH_LENGTH as u64;
    let mut input = RandomInput::new(BENCH_LENGTH);
    b.iter(|| bao_parallel_parents(input.get(), CHUNK_SIZE));
}

#[bench]
fn bench_bao_parallel_parents_3_large(b: &mut Bencher) {
    b.bytes = BENCH_LENGTH as u64;
    let mut input = RandomInput::new(BENCH_LENGTH);
    b.iter(|| bao_parallel_parents(input.get(), LARGE_CHUNK_SIZE));
}

#[bench]
fn bench_bao_large_chunks(b: &mut Bencher) {
    b.bytes = BENCH_LENGTH as u64;
    let mut input = RandomInput::new(BENCH_LENGTH);
    b.iter(|| bao_large_chunks(input.get()));
}

#[bench]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_load_8_blake2s_blocks_simple(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
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
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_load_8_blake2s_blocks_interleave(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
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
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_load_8_blake2s_blocks_gather(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
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
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_load_8_blake2s_blocks_gather_inner(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let blocks = avx2_blake2s_load::random_8_blocks();
    let mut out = unsafe { mem::zeroed() };
    b.iter(|| unsafe {
        avx2_blake2s_load::gather_from_blocks(&blocks, &mut out);
        test::black_box(&mut out);
    })
}

#[bench]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_load_4_blake2b_blocks_simple(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
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
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_load_4_blake2b_blocks_interleave(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
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
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_load_4_blake2b_blocks_gather(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
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
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_load_4_blake2b_blocks_gather_inner(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let blocks = avx2_blake2b_load::random_4_blocks();
    let mut out = unsafe { mem::zeroed() };
    b.iter(|| unsafe {
        avx2_blake2b_load::gather_from_blocks(&blocks, &mut out);
        test::black_box(&mut out);
    })
}
