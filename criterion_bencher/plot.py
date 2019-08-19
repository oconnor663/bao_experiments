#! /usr/bin/env python3

import json
from matplotlib import pyplot
from pathlib import Path
import pandas
import seaborn
import sys

HASH_NAMES = [
    ("bao", "Bao"),
    ("blake2s", "BLAKE2s"),
    ("blake2b", "BLAKE2b"),
    ("blake2sp", "BLAKE2sp"),
    ("blake2bp", "BLAKE2bp"),
    ("sha256", "OpenSSL SHA-256"),
    ("sha512", "OpenSSL SHA-512"),
]

SIZES = [
    (2**6, "64"),
    (2**7, "128"),
    (2**8, "256"),
    (2**9, "512"),
    (2**10, "1 KiB"),
    (2**11, "2 KiB"),
    (2**12, "4 KiB"),
    (2**13, "8 KiB"),
    (2**14, "16 KiB"),
    (2**15, "32 KiB"),
    (2**16, "64 KiB"),
    (2**17, "128 KiB"),
    (2**18, "256 KiB"),
    (2**19, "512 KiB"),
    (2**20, "1 MiB"),
]


def main():
    target = Path(sys.argv[1])
    title = target.with_suffix(".title").open().read().strip()
    columns = ["function", "size", "throughput"]
    data = []
    for hash_name, hash_name_pretty in HASH_NAMES:
        hash_dir = target / "bench_group" / hash_name
        for size, size_pretty in SIZES:
            estimates_path = hash_dir / str(size) / "new/estimates.json"
            estimates = json.load(estimates_path.open())
            slope = estimates["Slope"]
            point = slope["point_estimate"]
            # upper = slope["confidence_interval"]["upper_bound"]
            # lower = slope["confidence_interval"]["lower_bound"]
            mbps_throughput = size / point * 1000
            data.append([hash_name_pretty, size_pretty, mbps_throughput])
    dataframe = pandas.DataFrame(data, columns=columns)

    seaborn.set()
    pyplot.figure(figsize=[20, 10])
    seaborn.set_context("talk")
    plot = seaborn.barplot(data=dataframe,
                           x="size",
                           y="throughput",
                           hue="function")
    plot.set(xlabel="input bytes",
             ylabel="throughput (MB/s)",
             title=title)
    # plot.set_xticklabels(rotation=30)
    # pyplot.savefig("out.svg")
    pyplot.show()


if __name__ == "__main__":
    main()
