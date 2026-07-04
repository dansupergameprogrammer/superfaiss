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

SCHEMA_VERSION = 1        # emitted for channel-less banks
SCHEMA_VERSION_CHANNELS = 2  # emitted when a channels table is present
KNOWN_SCHEMA_VERSIONS = (1, 2)
METRICS = ("dot", "cosine", "l2")
MAX_CHANNELS = 8


def _validate_channels(channels, dims):
    if not channels:
        raise ValueError("schemaVersion 2 requires a non-empty channels table")
    if len(channels) > MAX_CHANNELS:
        raise ValueError(f"{len(channels)} channels, max {MAX_CHANNELS}")
    names = set()
    prev_end = 0
    for i, ch in enumerate(channels):
        name = ch.get("name")
        offset = ch.get("offset")
        length = ch.get("dims")
        if not name or not isinstance(offset, int) or not isinstance(length, int):
            raise ValueError(f"channel {i} needs name, offset, dims")
        if name in names:
            raise ValueError(f"duplicate channel name {name!r}")
        if offset < prev_end or length <= 0 or offset + length > dims:
            raise ValueError(f"channel {name!r} violates ascending/bounds rules")
        # Grid alignment (16-byte element grid) is quantization-dependent and is
        # enforced by the importer at bake; the interchange rule checked here is
        # structural (ascending, non-overlapping, in bounds, unique names).
        names.add(name)
        prev_end = offset + length
    return channels


def write_bank(name, rows, dims, metric="cosine", ids=None, description="", channels=None):
    """rows: iterable of iterables of float, all length dims. ids: optional list of str.
    channels: optional list of {"name", "offset", "dims"} dicts -> schemaVersion 2."""
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
        "schemaVersion": SCHEMA_VERSION_CHANNELS if channels else SCHEMA_VERSION,
        "dims": dims,
        "count": count,
        "metric": metric,
        "dtype": "float32",
        "description": description,
    }
    if ids is not None:
        header["ids"] = list(ids)
    if channels:
        header["channels"] = _validate_channels(list(channels), dims)
    with open(name + ".wvbank.json", "w", encoding="utf-8") as f:
        json.dump(header, f, indent=1)
    return count


def read_bank(json_path):
    """Returns (header, rows) where rows is a list of float lists."""
    with open(json_path, "r", encoding="utf-8") as f:
        header = json.load(f)
    if header.get("schemaVersion") not in KNOWN_SCHEMA_VERSIONS:
        raise ValueError(f"unknown schemaVersion {header.get('schemaVersion')}")
    if header.get("dtype") != "float32":
        raise ValueError(f"unknown dtype {header.get('dtype')}")
    dims = int(header["dims"])
    if header.get("schemaVersion") == SCHEMA_VERSION_CHANNELS:
        _validate_channels(header.get("channels"), dims)
    elif header.get("channels"):
        raise ValueError("channels table requires schemaVersion 2")
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
