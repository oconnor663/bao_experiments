#![feature(test)]

extern crate bao_experiments;
extern crate test;

use test::Bencher;

// The 4ary implementation is only defined for inputs that are a power of 4 times the chunk size.
const LENGTH: usize = 1 << 24; // about 17 MB

fn input(b: &mut Bencher, size: usize) -> Vec<u8> {
    b.bytes = size as u64;
    vec![0; size]
}

#[bench]
fn bench_blake2s(b: &mut Bencher) {
    let input = input(b, LENGTH);
    b.iter(|| bao_experiments::bao_blake2s(&input));
}

#[bench]
fn bench_blake2b_standard(b: &mut Bencher) {
    let input = input(b, LENGTH);
    b.iter(|| bao_experiments::bao_standard(&input));
}

#[bench]
fn bench_blake2b_standard_parallel_parents(b: &mut Bencher) {
    let input = input(b, LENGTH);
    b.iter(|| bao_experiments::bao_standard_parallel_parents(&input));
}

#[bench]
fn bench_blake2b_4ary(b: &mut Bencher) {
    let input = input(b, LENGTH);
    b.iter(|| bao_experiments::bao_4ary(&input));
}

#[bench]
fn bench_blake2b_4ary_parallel_parents(b: &mut Bencher) {
    let input = input(b, LENGTH);
    b.iter(|| bao_experiments::bao_4ary_parallel_parents(&input));
}

#[bench]
fn bench_blake2b_large_chunks(b: &mut Bencher) {
    let input = input(b, LENGTH);
    b.iter(|| bao_experiments::bao_blake2b_large_chunks(&input));
}

// NOTE: This benchmark is slower than it should be, for lack of an SSE implementation of BLAKE2s.
#[bench]
fn bench_blake2hybrid(b: &mut Bencher) {
    let input = input(b, LENGTH);
    b.iter(|| bao_experiments::bao_blake2hybrid(&input));
}

#[bench]
fn bench_blake2hybrid_parallel_parents(b: &mut Bencher) {
    let input = input(b, LENGTH);
    b.iter(|| bao_experiments::bao_blake2hybrid_parallel_parents(&input));
}
