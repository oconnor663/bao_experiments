#! /usr/bin/env python3

import json
from matplotlib import pyplot
from pathlib import Path
import pandas
import seaborn
import sys

HASH_NAMES = {
    "bao": "Bao",
    "blake2b": "BLAKE2b",
    "blake2bp": "BLAKE2bp",
    "blake2s": "BLAKE2s",
    "blake2sp": "BLAKE2sp",
    "sha256": "OpenSSL SHA-256",
    "sha512": "OpenSSL SHA-512",
}

SIZES = {
    2**6: "64",
    2**7: "128",
    2**8: "256",
    2**9: "512",
    2**10: "1 KiB",
    2**11: "2 KiB",
    2**12: "4 KiB",
    2**13: "8 KiB",
    2**14: "16 KiB",
    2**15: "32 KiB",
    2**16: "64 KiB",
    2**17: "128 KiB",
    2**18: "256 KiB",
    2**19: "512 KiB",
    2**20: "1 MiB",
}


def main():
    target = Path(sys.argv[1])
    index = [item[1] for item in sorted(SIZES.items())]
    column_names = []
    rows = [[] for _ in index]
    for hash_name, hash_name_pretty in sorted(HASH_NAMES.items()):
        column_names.append(hash_name_pretty)
        hash_dir = target / "bench_group" / hash_name
        for (size, size_pretty), row in zip(sorted(SIZES.items()), rows):
            estimates_path = hash_dir / str(size) / "new/estimates.json"
            estimates = json.load(estimates_path.open())
            slope = estimates["Slope"]
            point = slope["point_estimate"]
            # upper = slope["confidence_interval"]["upper_bound"]
            # lower = slope["confidence_interval"]["lower_bound"]
            mbps_throughput = size / point * 1000
            row.append(mbps_throughput)
    assert len(rows) == len(index), "wrong number of rows"
    for row in rows:
        assert len(row) == len(column_names), "wrong number of columns"
    dataframe = pandas.DataFrame(rows, index=index, columns=column_names)

    seaborn.set()
    plot = seaborn.relplot(data=dataframe)
    plot.set(xlabel="input bytes",
             ylabel="throughput (MB/s)",
             title="hash function throughput")
    # plot.set_xticklabels(rotation=30)
    pyplot.show()


if __name__ == "__main__":
    main()
