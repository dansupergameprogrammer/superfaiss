#!/usr/bin/env python3
"""Coherence check 3 — doc signature.

Every function signature documented in the vendored core's docs/API.md must
match the header that declares it. API.md organizes its content under `##
<header>.h — ...` section headings, each followed by a fenced ```cpp block
that is the section's declared API (later blocks in a section, if any, are
usage examples and are skipped). This script parses the FIRST fenced block
under each heading, extracts every function declaration in it, and confirms
the same declaration (return type + parameter TYPES, names ignored) exists
in the header(s) the heading names.

Run from anywhere; pass --vendored-root pointing at the vendored core tree
(the directory containing docs/API.md and include/superfaiss/).

Exit codes: 0 = every documented signature matched; 1 = at least one
mismatch, OR zero signatures were found to check (a check that can pass by
finding nothing to check is not a check).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _cxx_decls import extract_declarations  # noqa: E402

HEADING_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)
HEADER_TOKEN_RE = re.compile(r"(\w[\w]*\.h)\b")
CPP_BLOCK_RE = re.compile(r"```cpp\n(.*?)```", re.DOTALL)


def find_sections(api_md_text: str) -> list[tuple[str, list[str], str]]:
    """Returns (heading text, header filenames named in it, first cpp block)."""
    headings = list(HEADING_RE.finditer(api_md_text))
    sections = []
    for idx, m in enumerate(headings):
        heading = m.group(1)
        headers = HEADER_TOKEN_RE.findall(heading)
        if not headers:
            continue
        start = m.end()
        end = headings[idx + 1].start() if idx + 1 < len(headings) else len(api_md_text)
        body = api_md_text[start:end]
        block_m = CPP_BLOCK_RE.search(body)
        if not block_m:
            continue
        sections.append((heading, headers, block_m.group(1)))
    return sections


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendored-root", required=True, type=Path)
    args = ap.parse_args()

    api_md_path = args.vendored_root / "docs" / "API.md"
    include_dir = args.vendored_root / "include" / "superfaiss"
    if not api_md_path.is_file():
        print(f"FAIL: doc-signature check found no file at {api_md_path}")
        return 1
    if not include_dir.is_dir():
        print(f"FAIL: doc-signature check found no header directory at {include_dir}")
        return 1

    sections = find_sections(api_md_path.read_text(encoding="utf-8"))
    if not sections:
        print("FAIL: doc-signature check parsed zero API.md sections — "
              "nothing was checked. This is a failure, not a pass.")
        return 1

    total_checked = 0
    mismatches: list[str] = []

    header_cache: dict[str, dict[str, list]] = {}

    def decls_for_header(header_name: str) -> dict[str, list]:
        if header_name in header_cache:
            return header_cache[header_name]
        path = include_dir / header_name
        if not path.is_file():
            header_cache[header_name] = {}
            return {}
        decls = extract_declarations(path.read_text(encoding="utf-8"), has_namespace_wrapper=True)
        by_name: dict[str, list] = {}
        for d in decls:
            by_name.setdefault(d.name, []).append(d)
        header_cache[header_name] = by_name
        return by_name

    for heading, headers, block in sections:
        doc_decls = extract_declarations(block, has_namespace_wrapper=False)
        if not doc_decls:
            continue
        combined: dict[str, list] = {}
        missing_headers = [h for h in headers if not (include_dir / h).is_file()]
        for h in missing_headers:
            mismatches.append(
                f"[{heading}] documents against header '{h}' which does not exist "
                f"at {include_dir / h}"
            )
        for h in headers:
            for name, ds in decls_for_header(h).items():
                combined.setdefault(name, []).extend(ds)

        for d in doc_decls:
            total_checked += 1
            candidates = combined.get(d.name, [])
            if not candidates:
                mismatches.append(
                    f"[{heading}] docs/API.md declares `{d.shape_str()}` — "
                    f"no declaration named '{d.name}' found in {', '.join(headers)}"
                )
                continue
            if d.params == ["..."]:
                # An explicit `(...)` in API.md is the doc's own convention for
                # "elided here, see the header" (used for the lower-priority
                # parallel/segmented kernel variants) — confirm the name exists,
                # not the full parameter list it deliberately did not restate.
                continue
            if not any(c.shape() == d.shape() for c in candidates):
                best = candidates[0]
                mismatches.append(
                    f"[{heading}] docs/API.md declares `{d.shape_str()}` — "
                    f"header declares `{best.shape_str()}` (from {', '.join(headers)}) — signature mismatch"
                )

    if total_checked == 0:
        print("FAIL: doc-signature check extracted zero documented signatures — "
              "nothing was checked. This is a failure, not a pass.")
        return 1

    if mismatches:
        print(f"FAIL: doc-signature check — {len(mismatches)} of {total_checked} "
              f"documented signature(s) do not match their header:")
        for m in mismatches:
            print(f"  - {m}")
        return 1

    print(f"OK: doc-signature check — {total_checked} documented signature(s) "
          f"matched their declaring header.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
