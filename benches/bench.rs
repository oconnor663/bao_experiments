#![feature(test)]

extern crate bao_experiments;
extern crate test;

use bao_experiments::Finalization::*;
use test::Bencher;

const LENGTH: usize = 10_000_000;

fn input(b: &mut Bencher, size: usize) -> Vec<u8> {
    b.bytes = size as u64;
    vec![0; size]
}

#[bench]
fn bench_standard(b: &mut Bencher) {
    let input = input(b, LENGTH);
    b.iter(|| bao_experiments::hash_recurse_rayon_blake2b(&input, Root(LENGTH as u64)));
}

#[bench]
fn bench_blake2s(b: &mut Bencher) {
    let input = input(b, LENGTH);
    b.iter(|| bao_experiments::hash_recurse_rayon_blake2s(&input, Root(LENGTH as u64)));
}
