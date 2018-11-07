This repo currently contains several experiments, suggested by
https://twitter.com/zooko/status/1058628340918050816. They're all aimed
at reducing the overhead of hashing parent nodes:

- Use BLAKE2s instead of BLAKE2b.
- Use a hybrid of BLAKE2b for chunks and BLAKE2s for parents.
- Use a 4-ary tree.
- Use a larger chunk size.

For now, building this repo requires you to clone these two adjacent to
it:

- https://github.com/oconnor663/blake2b_simd
- https://github.com/oconnor663/blake2s_simd

Run the benchmarks with `cargo +nightly bench`. This executes the same
benchmarks with `libtest` and with
[`criterion`](https://github.com/japaric/criterion.rs). Note that the
`libtest` benchmarks vary by as much as 40% depending on whether the
processor is already hot, because they don't do their own warmups.
