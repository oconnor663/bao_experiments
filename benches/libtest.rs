#![feature(test)]

extern crate bao_experiments;
extern crate rand;
extern crate test;

use bao_experiments::*;
use rand::seq::SliceRandom;
use rand::RngCore;
use std::mem;
use test::Bencher;

// This struct randomizes two things:
// 1. The actual bytes of input.
// 2. The page offset the input starts at.
pub struct RandomInput {
    buf: Vec<u8>,
    len: usize,
    offsets: Vec<usize>,
    offset_index: usize,
}

impl RandomInput {
    pub fn new(len: usize) -> Self {
        let page_size: usize = page_size::get();
        let mut buf = vec![0u8; len + page_size];
        let mut rng = rand::thread_rng();
        rng.fill_bytes(&mut buf);
        let mut offsets: Vec<usize> = (0..page_size).collect();
        offsets.shuffle(&mut rng);
        Self {
            buf,
            len,
            offsets,
            offset_index: 0,
        }
    }

    pub fn get(&mut self) -> &[u8] {
        let offset = self.offsets[self.offset_index];
        self.offset_index += 1;
        if self.offset_index >= self.offsets.len() {
            self.offset_index = 0;
        }
        &self.buf[offset..][..self.len]
    }
}

#[bench]
fn bench_bao_standard(b: &mut Bencher) {
    b.bytes = BENCH_LENGTH as u64;
    let mut input = RandomInput::new(BENCH_LENGTH);
    b.iter(|| bao_standard(input.get()));
}

#[bench]
fn bench_bao_parallel_parents(b: &mut Bencher) {
    b.bytes = BENCH_LENGTH as u64;
    let mut input = RandomInput::new(BENCH_LENGTH);
    b.iter(|| bao_parallel_parents(input.get()));
}

#[bench]
fn bench_bao_evil_random(b: &mut Bencher) {
    b.bytes = BENCH_LENGTH as u64;
    let mut input = RandomInput::new(BENCH_LENGTH);
    b.iter(|| bao_evil(input.get()));
}

#[bench]
fn bench_bao_evil_zeros(b: &mut Bencher) {
    b.bytes = BENCH_LENGTH as u64;
    let input = vec![0; BENCH_LENGTH];
    let input = test::black_box(input).clone();
    b.iter(|| bao_evil(&input));
}

#[bench]
fn bench_bao_large_chunks(b: &mut Bencher) {
    b.bytes = BENCH_LENGTH as u64;
    let mut input = RandomInput::new(BENCH_LENGTH);
    b.iter(|| bao_large_chunks(input.get()));
}

#[bench]
fn bench_bao_nary(b: &mut Bencher) {
    b.bytes = BENCH_LENGTH as u64;
    let mut input = RandomInput::new(BENCH_LENGTH);
    b.iter(|| bao_nary(input.get()));
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
