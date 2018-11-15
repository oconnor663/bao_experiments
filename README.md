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
are without Turbo Boost.

```
64-bit i5-8250U multithreaded (4/8 cores)
-----------------------------------------
BLAKE2s large chunks               6.0138 GiB/s
BLAKE2s parallel parents           5.6268 GiB/s
BLAKE2b large chunks               5.5165 GiB/s
BLAKE2b 4-ary parallel parents     5.4020 GiB/s
BLAKE2b 4-ary                      5.3837 GiB/s
BLAKE2hybrid parallel parents      5.3676 GiB/s
BLAKE2s                            5.3216 GiB/s
BLAKE2b standard                   5.1867 GiB/s
BLAKE2b standard parallel parents  5.1051 GiB/s
BLAKE2hybrid                       5.0448 GiB/s

64-bit i5-8250U singlethreaded
-----------------------------------------
BLAKE2s large chunks               2.2979 GiB/s
BLAKE2s parallel parents           2.1907 GiB/s
BLAKE2s                            2.0264 GiB/s
BLAKE2b 4-ary parallel parents     1.9443 GiB/s
BLAKE2b 4-ary                      1.9138 GiB/s
BLAKE2b large chunks               1.9076 GiB/s
BLAKE2hybrid parallel parents      1.8033 GiB/s
BLAKE2hybrid                       1.7186 GiB/s
BLAKE2b standard parallel parents  1.7079 GiB/s
BLAKE2b standard                   1.6142 GiB/s

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
