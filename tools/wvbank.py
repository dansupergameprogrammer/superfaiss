#!/usr/bin/env python3
"""Reference reader/writer for the SuperFAISS .wvbank sidecar interchange format.

A bank on disk is two files:
  <name>.wvbank.json  -- header (this module defines the schema)
  <name>.wvbank.bin   -- raw float32 rows, row-major, count*dims*4 bytes, little-endian

The binary payload is always float32 and unpadded; padding, normalization (Cosine
banks), and int8 quantization happen at bake time in the importer, using the library's
bake functions. Keeping the interchange format this primitive is deliberate: any
pipeline can emit it in a few lines.

Usage:
  python wvbank.py write out_name vectors.npy --metric cosine --ids ids.txt
  python wvbank.py read name.wvbank.json
"""

import json
import struct
import sys

SCHEMA_VERSION = 1
METRICS = ("dot", "cosine", "l2")


def write_bank(name, rows, dims, metric="cosine", ids=None, description=""):
    """rows: iterable of iterables of float, all length dims. ids: optional list of str."""
    count = 0
    with open(name + ".wvbank.bin", "wb") as f:
        for row in rows:
            if len(row) != dims:
                raise ValueError(f"row {count} has {len(row)} dims, expected {dims}")
            f.write(struct.pack(f"<{dims}f", *row))
            count += 1
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    if ids is not None and len(ids) != count:
        raise ValueError(f"{len(ids)} ids for {count} rows")
    header = {
        "schemaVersion": SCHEMA_VERSION,
        "dims": dims,
        "count": count,
        "metric": metric,
        "dtype": "float32",
        "description": description,
    }
    if ids is not None:
        header["ids"] = list(ids)
    with open(name + ".wvbank.json", "w", encoding="utf-8") as f:
        json.dump(header, f, indent=1)
    return count


def read_bank(json_path):
    """Returns (header, rows) where rows is a list of float lists."""
    with open(json_path, "r", encoding="utf-8") as f:
        header = json.load(f)
    if header.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError(f"unknown schemaVersion {header.get('schemaVersion')}")
    if header.get("dtype") != "float32":
        raise ValueError(f"unknown dtype {header.get('dtype')}")
    dims = int(header["dims"])
    count = int(header["count"])
    if header.get("metric") not in METRICS:
        raise ValueError(f"unknown metric {header.get('metric')}")
    bin_path = json_path[: -len(".json")] + ".bin"
    rows = []
    with open(bin_path, "rb") as f:
        payload = f.read()
    expected = count * dims * 4
    if len(payload) != expected:
        raise ValueError(f"payload is {len(payload)} bytes, header implies {expected}")
    for r in range(count):
        rows.append(list(struct.unpack_from(f"<{dims}f", payload, r * dims * 4)))
    ids = header.get("ids")
    if ids is not None and len(ids) != count:
        raise ValueError(f"{len(ids)} ids for {count} rows")
    return header, rows


def _main(argv):
    if len(argv) >= 2 and argv[1] == "read":
        header, rows = read_bank(argv[2])
        print(json.dumps({k: v for k, v in header.items() if k != "ids"}, indent=1))
        print(f"rows: {len(rows)}")
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
