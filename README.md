This repo currently contains several experiments, suggested by
https://twitter.com/zooko/status/1058628340918050816. They're all aimed
at reducing the overhead of hashing parent nodes:

- Use BLAKE2s instead of BLAKE2b.
- Use a hybrid of BLAKE2b for chunks and BLAKE2s for parents.
- Use a 4-ary tree.
- Use a larger chunk size.
- Use `update4` (BLAKE2b) or `update8` (BLAKE2s) on parents. (We always
  use them on chunks.)

Run the benchmarks with `cargo +nightly bench`. This executes the same
benchmarks with `libtest` and with
[`criterion`](https://github.com/japaric/criterion.rs). Note that the
`libtest` benchmarks vary by as much as 40% depending on whether the
processor is already hot, because they don't do their own warmups.

Below are the current throughput averages from the criterion benchmarks,
arranged from fastest to slowest. Note that we're not using a SIMD
implementation of single-instance BLAKE2s yet, so the `BLAKE2s` and
`BLAKE2hybrid` benchmarks could be faster, but all the chunk hashing is
parallelized in any case.

```
BLAKE2b large chunks               5.2360 GiB/s
BLAKE2b 4-ary                      5.0212 GiB/s
BLAKE2b 4-ary parallel parents     4.9355 GiB/s
BLAKE2b standard                   4.8290 GiB/s
BLAKE2b standard parallel parents  4.7956 GiB/s
BLAKE2hybrid parallel parents      4.7929 GiB/s
BLAKE2hybrid                       4.7494 GiB/s
BLAKE2s                            4.2716 GiB/s
```

Takeaways:

- BLAKE2s doesn't seem to help. This is related to the
  `blake2s_simd::update8` function having about 7% less throughput than
  `blake2b_simd::update4`. Their code is very similar, with the
  differences being that `update8` uses `_mm256_add_epi32` instead of
  `_mm256_add_epi64`, that it loads and stores more words, and that it
  uses different rotations. It's possible that one of thoses differences
  hurts performance, or that I introduced some inefficiency somewhere
  accidentally. Notably, Samuel Neves' [C
  benchmarks](https://github.com/sneves/blake2-avx2/blob/master/bench.sh)
  run on my machine have BLAKE2sp slightly outperforming BLAKE2bp, while
  I'm getting the opposite from my Rust code.
- Hashing multiple parents in parallel on each thread also doesn't seem
  to help. Again, I'm not sure why. Maybe growing the working set of
  each thread hurts cache performance.
- A 4-ary tree layout is about 4% faster, due to making full use of the
  128-byte BLAKE2b block size. The implementations we're testing here
  are simplified, though, without the logic necessary to handle
  not-evenly-divisible input lengths, so it's possible a full
  implementation would be slightly slower on general inputs.
- The biggest improvement, about 8%, comes from increasing the chunk
  size, which reduces the total number of parent nodes. This benchmark
  uses a 65536-byte chunk size, rather than the 4096-byte size used in
  the standard implementation, so there are 16x fewer parent nodes in
  the large-chunk tree. These results are in line with previous
  experiments around chunk size; see the [discussion
  issue](https://github.com/oconnor663/bao/issues/17) about chunk size
  tradeoffs.
