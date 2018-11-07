This repo currently contains two experiments, both inspired by
https://twitter.com/zooko/status/1058628340918050816

1. What if Bao used BLAKE2s (`update8`) instead of BLAKE2b (`update4`)?
   This could reduce parent node overhead, because the BLAKE2s block
   size is exactly two parent nodes.
2. What if Bao used a 4-way tree instead of a binary tree? This could
   reduce parent node overhead, because the BLAKE2b block size is
   exactly four parent nodes.

For now, building this repo requires you to clone these two adjacent to
it:

- https://github.com/oconnor663/blake2b_simd
- https://github.com/oconnor663/blake2s_simd

Run the benchmarks with `cargo +nightly bench`. This executes the same
benchmarks with `libtest` and with
[`criterion`](https://github.com/japaric/criterion.rs).
