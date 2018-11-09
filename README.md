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
processor is already hot, because they don't do their own warmups. That
said, even the criterion benchmarks seem to have as much as 10% variance
from run to run.

Below are some preliminary numbers from my benchmark runs, arranged from
fastest to slowest. Note that the Intel runs use long warmups, so these numbers
are without Turbo Boost. Note also that we're not using a SIMD implementation
of single-instance BLAKE2s yet, so the `BLAKE2s` and `BLAKE2hybrid` benchmarks
on Intel will get faster.

```
64-bit i5-8250U multithreaded (4/8 cores)
-----------------------------------------
BLAKE2b large chunks               5.2360 GiB/s
BLAKE2b 4-ary                      5.0212 GiB/s
BLAKE2b 4-ary parallel parents     4.9355 GiB/s
BLAKE2b standard                   4.8290 GiB/s
BLAKE2b standard parallel parents  4.7956 GiB/s
BLAKE2hybrid parallel parents      4.7929 GiB/s
BLAKE2hybrid                       4.7494 GiB/s
BLAKE2s                            4.2716 GiB/s

64-bit i5-8250U singlethreaded
-----------------------------------------
BLAKE2b large chunks               1.7752 GiB/s
BLAKE2hybrid parallel parents      1.7181 GiB/s
BLAKE2b 4-ary parallel parents     1.6935 GiB/s
BLAKE2b 4-ary                      1.6804 GiB/s
BLAKE2hybrid                       1.6661 GiB/s
BLAKE2b standard                   1.5988 GiB/s
BLAKE2s                            1.4944 GiB/s
BLAKE2b standard parallel parents  1.4826 GiB/s

32-bit ARM v7l multithreaded (4 cores)
-------------------------------
BLAKE2s                            244 MB/s
BLAKE2b 4-ary                      132 MB/s
BLAKE2b standard                   130 MB/s
BLAKE2b 4-ary parallel parents     120 MB/s
BLAKE2hybrid                       116 MB/s
BLAKE2b standard parallel parents  106 MB/s
BLAKE2b large chunks                92 MB/s
BLAKE2hybrid parallel parents       86 MB/s

32-bit ARM v7l singlethreaded
-----------------------------
BLAKE2s                            65 MB/s
BLAKE2b large chunks               36 MB/s
BLAKE2b 4-ary parallel parents     35 MB/s
BLAKE2b 4-ary                      34 MB/s
BLAKE2b standard                   34 MB/s
BLAKE2b standard parallel parents  34 MB/s
BLAKE2hybrid parallel parents      34 MB/s
BLAKE2hybrid                       33 MB/s
```
