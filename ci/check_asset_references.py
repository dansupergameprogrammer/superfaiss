#!/usr/bin/env python3
"""Coherence check 4 — asset reference.

Every image/link path referenced by a Markdown file must exist at that exact
path, CASE-SENSITIVE. Windows and macOS filesystems are case-insensitive by
default, so a path that resolves locally (`docs/images/x.png` against a
`Docs/Images/x.png` on disk) can still 404 on GitHub, which serves case-
sensitively. This script never trusts `os.path.exists` for that reason — it
walks the real directory entries and compares names byte-for-byte.

Only relative, same-repo paths are checked (a leading `/`, a URL scheme, or
an in-page `#anchor` is out of scope — those are not files this repo owns).

Usage:
    check_asset_references.py --repo-root <path> [<markdown file>...]

    With no file arguments, every *.md file in --repo-root is scanned
    (excluding .git and common build-output directories).

Exit codes: 0 = every reference resolved with matching case; 1 = at least
one reference did not, OR zero references were found (a check that can pass
by finding nothing to check is not a check).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit, unquote

LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
EXCLUDE_DIR_NAMES = {".git", "Binaries", "Intermediate", "Saved", "DerivedDataCache", "node_modules"}


def find_markdown_files(repo_root: Path) -> list[Path]:
    out = []
    for p in repo_root.rglob("*.md"):
        if any(part in EXCLUDE_DIR_NAMES for part in p.relative_to(repo_root).parts):
            continue
        out.append(p)
    return sorted(out)


def case_sensitive_exists(path: Path) -> bool:
    """True iff every path component matches an on-disk entry's exact case."""
    if not path.exists():
        return False
    cur = path
    while True:
        parent = cur.parent
        if parent == cur:
            return True
        try:
            entries = {e.name for e in parent.iterdir()}
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            return False
        if cur.name not in entries:
            return False
        cur = parent


def extract_local_targets(md_path: Path) -> list[str]:
    text = md_path.read_text(encoding="utf-8", errors="replace")
    targets = []
    for m in LINK_RE.finditer(text):
        raw = m.group(1)
        parsed = urlsplit(raw)
        if parsed.scheme or raw.startswith("//"):
            continue  # external URL
        target = parsed.path
        if not target or target.startswith("/"):
            continue  # repo-absolute or anchor-only; not this repo's relative-path concern
        targets.append(unquote(target))
    return targets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True, type=Path)
    ap.add_argument("files", nargs="*", type=Path)
    args = ap.parse_args()

    # Deliberately os.path.abspath, never Path.resolve(): resolve() queries the
    # filesystem and silently rewrites a path to its ACTUAL on-disk case on
    # Windows/macOS, which would erase the exact defect this check exists to
    # catch. abspath is purely lexical (cwd + normpath) — it never touches disk.
    repo_root = Path(os.path.abspath(args.repo_root))
    md_files = [Path(os.path.abspath(f)) for f in args.files] if args.files else find_markdown_files(repo_root)

    if not md_files:
        print("FAIL: asset-reference check found zero Markdown files to scan — "
              "this is a failure, not a pass.")
        return 1

    total_refs = 0
    failures: list[str] = []

    for md in md_files:
        for target in extract_local_targets(md):
            total_refs += 1
            resolved = Path(os.path.normpath(md.parent / target))
            if not case_sensitive_exists(resolved):
                rel_md = md.relative_to(repo_root) if repo_root in md.parents or md == repo_root else md
                failures.append(f"{rel_md}: references '{target}' -> no case-exact match at {resolved}")

    if total_refs == 0:
        print("FAIL: asset-reference check found zero local file references across "
              f"{len(md_files)} Markdown file(s) — this is a failure, not a pass.")
        return 1

    if failures:
        print(f"FAIL: asset-reference check — {len(failures)} of {total_refs} reference(s) "
              f"did not resolve case-exactly:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"OK: asset-reference check — {total_refs} reference(s) across {len(md_files)} "
          f"Markdown file(s) all resolved case-exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
