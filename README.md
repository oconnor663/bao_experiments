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
fastest to slowest. These numbers are the median of three runs. Note
that the Intel numbers are with Turbo Boost turned off. If Turbo Boost
is turned on, the single threaded figures on my laptop are about twice
as high, even for long runs with a hot CPU. The boost for multithreaded
benchmarks is smaller, about 33%.

```
64-bit i5-8250U multithreaded (4/8 cores) Turbo-Boost-disabled
--------------------------------------------------------------
BLAKE2s large chunks               4.51   GiB/s
BLAKE2s parallel parents           4.34   GiB/s
BLAKE2b large chunks               4.06   GiB/s
BLAKE2b 4-ary parallel parents     4.03   GiB/s
BLAKE2s                            4.03   GiB/s
BLAKE2b 4-ary                      4.00   GiB/s
BLAKE2hybrid parallel parents      3.98   GiB/s
BLAKE2b standard                   3.81   GiB/s
BLAKE2hybrid                       3.75   GiB/s
BLAKE2b standard parallel parents  3.71   GiB/s

64-bit i5-8250U singlethreaded Turbo-Boost-disabled
---------------------------------------------------
BLAKE2s large chunks               1.06   GiB/s
BLAKE2s parallel parents           1.03   GiB/s
BLAKE2b large chunks               0.98   GiB/s
BLAKE2s                            0.97   GiB/s
BLAKE2hybrid parallel parents      0.93   GiB/s
BLAKE2b 4-ary parallel parents     0.90   GiB/s
BLAKE2b standard parallel parents  0.90   GiB/s
BLAKE2hybrid                       0.89   GiB/s
BLAKE2b standard                   0.88   GiB/s
BLAKE2b 4-ary                      0.88   GiB/s

32-bit ARM v7l multithreaded (4 cores)
--------------------------------------
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
