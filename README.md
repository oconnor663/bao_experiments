This repo previously contained a number of experiments comparing BLAKE2b
and BLAKE2s. Eventually I figured out that BLAKE2s is faster in
essentially all cases. Now the repo contains some new experiments:

- Using hash_many on only leaves (`bao_standard`) vs using it on parents
  also (`bao_parallel_parents`).
- Using a very large chunk size (`bao_large_chunks`). This is a baseline
  for what performance should be if parent node overhead disappears
  entirely.
- Using a tree of arity > 2 (`bao_nary`).

Mostly unrelated to the above, it also contains some benchmarks for
different SIMD algorithms for loading transposed message words.

Run these benchmarks with `cargo +nightly bench`.
