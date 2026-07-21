"""Shared C++ declaration parsing used by check_doc_signatures.py.

Extracts function-declaration signatures from a chunk of C++ text (either a
header file or a fenced ```cpp block copied into documentation) and reduces
each one to a comparable, name-independent shape: return type, function
name, and parameter TYPES (parameter identifiers are dropped, since a header
and a prose copy of it are free to name a parameter differently without the
signatures actually disagreeing).

This is intentionally scoped to free functions and class/struct member
function declarations one level of nesting deep (i.e. declared directly
inside a `class`/`struct` body, or directly inside a single wrapping
`namespace`). It does not attempt to parse templates, operator overloads, or
function bodies — none of those shapes appear in this project's headers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


@dataclass
class Decl:
    name: str
    ret: str
    params: list[str] = field(default_factory=list)
    is_const_method: bool = False
    raw: str = ""

    def key(self) -> str:
        return self.name

    def shape(self) -> tuple:
        return (self.ret, tuple(self.params), self.is_const_method)

    def shape_str(self) -> str:
        params = ", ".join(self.params)
        suffix = " const" if self.is_const_method else ""
        return f"{self.ret} {self.name}({params}){suffix}"


_QUALIFIERS = ("static", "inline", "explicit", "virtual", "constexpr", "friend")


def _split_top_level_commas(s: str) -> list[str]:
    parts = []
    depth = 0
    cur = []
    for ch in s:
        if ch in "([<":
            depth += 1
        elif ch in ")]>":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return parts


def _normalize_param(p: str) -> str | None:
    p = p.strip()
    if not p:
        return None
    # Drop a default value.
    p = re.sub(r"=.*$", "", p).strip()
    if not p or p == "void":
        return None
    # Drop a trailing parameter identifier, if present (a bare type has none).
    m = re.match(r"^(.*[\*\&\s])([A-Za-z_]\w*)$", p)
    if m and m.group(2) not in ("const",):
        p = m.group(1)
    return re.sub(r"\s+", "", p)


def _parse_statement(stmt: str) -> Decl | None:
    stmt = re.sub(r"\s+", " ", stmt).strip()
    if not stmt or "(" not in stmt or ")" not in stmt:
        return None
    is_const_method = False
    if re.search(r"\)\s*const\s*$", stmt):
        is_const_method = True
        stmt = re.sub(r"\)\s*const\s*$", ")", stmt)
    if not stmt.endswith(")"):
        return None
    # Split at the LAST top-level '(' that closes at the very end — walk
    # from the end to find the matching '(' for the final ')'.
    depth = 0
    open_idx = None
    for i in range(len(stmt) - 1, -1, -1):
        ch = stmt[i]
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
            if depth == 0:
                open_idx = i
                break
    if open_idx is None:
        return None
    head = stmt[:open_idx].strip()
    params_str = stmt[open_idx + 1 : -1].strip()

    m = re.match(r"^(.*?)\b([A-Za-z_]\w*)$", head)
    if not m:
        return None
    ret_type, name = m.groups()
    ret_type = ret_type.strip()
    for kw in _QUALIFIERS:
        ret_type = re.sub(rf"(?:^|\s){kw}\b", " ", ret_type).strip()
    ret_type = re.sub(r"\s+", "", ret_type)

    if params_str == "" or params_str == "void":
        params = []
    else:
        params = [
            _normalize_param(p) for p in _split_top_level_commas(params_str)
        ]
        params = [p for p in params if p is not None]

    if not name or name in ("if", "for", "while", "switch", "return"):
        return None

    return Decl(name=name, ret=ret_type, params=params, is_const_method=is_const_method, raw=stmt)


_SIG_HEAD_RE = re.compile(r"\)\s*(const)?\s*$")


def extract_declarations(text: str, *, has_namespace_wrapper: bool) -> list[Decl]:
    """Extract candidate function declarations at brace-depth 0 (free
    functions / functions declared directly inside a namespace) or 1
    (members declared directly inside a class/struct/nested-namespace body).

    Two shapes are recognized: a plain forward declaration ending in `;`,
    and an inline-defined member (`int32_t Foo() const { return X; }`) whose
    body is skipped once the signature ahead of the `{` has been captured.
    """
    text = strip_comments(text)
    # Access-specifier labels ("public:") are not statements; drop them so they
    # cannot fuse onto the declaration that follows across a line break.
    text = re.sub(r"\b(?:public|private|protected)\s*:", " ", text)
    # A wrapping `namespace X { ... }` adds one real brace level around every
    # top-level declaration; a documentation code block never repeats that
    # wrapper, so its declarations sit one level shallower. Rather than
    # rewrite the text, shift the capture window instead.
    capture_depths = (1, 2) if has_namespace_wrapper else (0, 1)

    decls: list[Decl] = []
    depth = 0
    stmt_start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "{":
            pending = text[stmt_start:i].strip()
            # A constructor initializer list ("Foo(...) : a_(a), b_(b)") sits
            # between the real signature and the body; keep only the signature.
            ctor_m = re.search(r"\)(?=\s*:\s*[A-Za-z_])", pending)
            if ctor_m:
                pending = pending[: ctor_m.end()]
            if depth in capture_depths and pending and _SIG_HEAD_RE.search(pending):
                d = _parse_statement(pending)
                if d is not None:
                    decls.append(d)
                # Skip the inline body entirely.
                bdepth = 0
                j = i
                while j < n:
                    if text[j] == "{":
                        bdepth += 1
                    elif text[j] == "}":
                        bdepth -= 1
                        if bdepth == 0:
                            break
                    j += 1
                i = j + 1
                stmt_start = i
                continue
            depth += 1
            stmt_start = i + 1
        elif ch == "}":
            depth -= 1
            stmt_start = i + 1
        elif ch == ";":
            if depth in capture_depths:
                stmt = text[stmt_start:i]
                d = _parse_statement(stmt)
                if d is not None:
                    decls.append(d)
            stmt_start = i + 1
        i += 1
    return decls
