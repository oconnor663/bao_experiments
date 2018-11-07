This repo currently contains two experiments, both inspired by
https://twitter.com/zooko/status/1058628340918050816

1. What if Bao used BLAKE2s (`update8`) instead of BLAKE2b (`update4`)?
2. What if Bao used a 4-way tree instead of a binary tree, to cut down
   on parent node overhead?

For now, building this repo requires you to clone these two adjacent to
it:

- https://github.com/oconnor663/blake2b_simd
- https://github.com/oconnor663/blake2s_simd

Run the benchmarks with:

```
cargo +nightly bench
```
