#!/usr/bin/env python3
"""Coherence check — reverse doc signature (undocumented public symbol).

`check_doc_signatures.py` is one-directional: it confirms every signature
*documented* in docs/API.md matches the header that declares it, but never
asks the converse question — whether every symbol a public header *declares*
is documented anywhere. A newly-added free function can therefore ship
undocumented and CI stays green, because nothing on the documented side ever
pointed at it.

This check closes that gap. For every `include/superfaiss/*.h` header, it
extracts every free function declared directly in `namespace superfaiss`
(one level deep — not a struct/class member, and not anything inside a
nested `namespace detail`, which is the project's own convention for
internal-only helpers). It then confirms each one's name appears in some
fenced ```cpp block in docs/API.md.

This is a name-presence check, not a signature-equality check (that
direction is already covered by check_doc_signatures.py) — it exists to
catch a symbol that is missing from the docs entirely, which is a stronger
and structurally different failure than a signature that merely drifted.

Run from anywhere; pass --vendored-root pointing at the vendored core tree
(the directory containing docs/API.md and include/superfaiss/).

Exit codes: 0 = every declared free function is documented somewhere in
API.md; 1 = at least one is missing, OR zero declared free functions were
found to check (a check that can pass by finding nothing to check is not a
check).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _cxx_decls import extract_declarations, strip_comments  # noqa: E402

CPP_BLOCK_RE = re.compile(r"```cpp\n(.*?)```", re.DOTALL)
# A small helper is often documented as a call-syntax mention in prose rather
# than repeated as its own fenced declaration (e.g. `` `PaddedDims(dims, quant)` ``).
# That is still documentation — a name-presence check should recognize it.
INLINE_CALL_RE = re.compile(r"`([A-Za-z_]\w*)\(")

_STRUCT_OPEN_RE = re.compile(r"\b(?:struct|class)\s+[A-Za-z_]\w*\b[^{;]*(?=\{)")
_DETAIL_NS_OPEN_RE = re.compile(r"\bnamespace\s+detail\s*(?=\{)")


def _strip_braced_blocks(text: str, opener_re: re.Pattern) -> str:
    """Removes every `<opener> { ... } [;]` span the pattern's start matches,
    by brace-counting from the first `{` after the match to its close. Used
    to strip struct/class bodies and the `namespace detail { ... }` block so
    only free functions declared directly in `namespace superfaiss` remain
    at the depth `extract_declarations` treats as top-level.
    """
    while True:
        m = opener_re.search(text)
        if not m:
            return text
        brace_start = text.find("{", m.end())
        if brace_start == -1:
            # Malformed / no body found — drop just the opener to avoid looping.
            text = text[: m.start()] + text[m.end() :]
            continue
        depth = 0
        i = brace_start
        n = len(text)
        while i < n:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
        end = i
        j = end
        while j < n and text[j] in " \t\r\n":
            j += 1
        if j < n and text[j] == ";":
            end = j + 1
        text = text[: m.start()] + text[end:]


def free_function_names(header_text: str) -> set[str]:
    text = strip_comments(header_text)
    text = _strip_braced_blocks(text, _STRUCT_OPEN_RE)
    text = _strip_braced_blocks(text, _DETAIL_NS_OPEN_RE)
    decls = extract_declarations(text, has_namespace_wrapper=True)
    return {d.name for d in decls}


def documented_names(api_md_text: str) -> set[str]:
    names: set[str] = set()
    for block_m in CPP_BLOCK_RE.finditer(api_md_text):
        for d in extract_declarations(block_m.group(1), has_namespace_wrapper=False):
            names.add(d.name)
    # Also credit an inline call-syntax mention in prose outside a fenced block
    # (this project's convention for documenting small helpers without
    # repeating a full declaration).
    names.update(INLINE_CALL_RE.findall(api_md_text))
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendored-root", required=True, type=Path)
    args = ap.parse_args()

    api_md_path = args.vendored_root / "docs" / "API.md"
    include_dir = args.vendored_root / "include" / "superfaiss"
    if not api_md_path.is_file():
        print(f"FAIL: reverse-doc-signature check found no file at {api_md_path}")
        return 1
    if not include_dir.is_dir():
        print(f"FAIL: reverse-doc-signature check found no header directory at {include_dir}")
        return 1

    documented = documented_names(api_md_path.read_text(encoding="utf-8"))

    headers = sorted(include_dir.glob("*.h"))
    if not headers:
        print("FAIL: reverse-doc-signature check found zero headers — "
              "nothing was checked. This is a failure, not a pass.")
        return 1

    total_checked = 0
    missing: list[str] = []
    for header in headers:
        names = free_function_names(header.read_text(encoding="utf-8"))
        for name in sorted(names):
            total_checked += 1
            if name not in documented:
                missing.append(f"{header.name}: `{name}` is declared but not documented in docs/API.md")

    if total_checked == 0:
        print("FAIL: reverse-doc-signature check found zero declared free functions — "
              "nothing was checked. This is a failure, not a pass.")
        return 1

    if missing:
        print(f"FAIL: reverse-doc-signature check — {len(missing)} of {total_checked} "
              f"declared public free function(s) are undocumented:")
        for m in missing:
            print(f"  - {m}")
        return 1

    print(f"OK: reverse-doc-signature check — {total_checked} declared public free "
          f"function(s) across {len(headers)} header(s) are all documented in docs/API.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
