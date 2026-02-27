"""
ast_map.py — AST-based codebase map generator.

Extracts function/class signatures from all project Python files
and prints a structured, readable map without implementation noise.

Usage:
    python ast_map.py                     # full map, all directories
    python ast_map.py src                 # only src/ directory
    python ast_map.py src approximated    # multiple directories
    python ast_map.py --json              # output as JSON
    python ast_map.py --grep morans_i     # find a specific name
"""

import ast
import sys
import json
import pathlib
import textwrap
import argparse
from typing import Optional

# ── Configuration ────────────────────────────────────────────────────────────

# Directories to scan (relative to project root) — in display order
DEFAULT_DIRS = [
    "src",
    "approximated",
    "math_statistics",
    "tests",
    "raw_data/pdf_excel_data",
    ".",        # root-level scripts (main.py, build_soil_db.py, …)
]

# Directories / patterns to always skip
SKIP_DIRS = {".venv", "__pycache__", ".git", "node_modules"}

# Root of the project (the directory that contains this script)
ROOT = pathlib.Path(__file__).parent.resolve()

# ── AST helpers ──────────────────────────────────────────────────────────────

def _annotation_str(node: Optional[ast.expr]) -> str:
    """Convert an annotation AST node to its source string."""
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return "?"


def _arg_str(arg: ast.arg) -> str:
    """Format a single function argument with its annotation."""
    s = arg.arg
    ann = _annotation_str(arg.annotation)
    if ann:
        s += f": {ann}"
    return s


def _default_str(node: ast.expr) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "..."


def _func_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Reconstruct a human-readable function signature from AST."""
    args = node.args

    # positional args (no defaults)
    n_defaults = len(args.defaults)
    n_args = len(args.args)
    positional = []
    for i, arg in enumerate(args.args):
        default_offset = i - (n_args - n_defaults)
        if default_offset >= 0:
            default = f" = {_default_str(args.defaults[default_offset])}"
        else:
            default = ""
        positional.append(_arg_str(arg) + default)

    # *args
    vararg = [f"*{_arg_str(args.vararg)}"] if args.vararg else (["*"] if args.kwonlyargs else [])

    # keyword-only args
    kwonly = []
    for i, arg in enumerate(args.kwonlyargs):
        kw_default = args.kw_defaults[i]
        default = f" = {_default_str(kw_default)}" if kw_default is not None else ""
        kwonly.append(_arg_str(arg) + default)

    # **kwargs
    kwarg = [f"**{_arg_str(args.kwarg)}"] if args.kwarg else []

    all_parts = positional + vararg + kwonly + kwarg
    params = ", ".join(all_parts)

    ret = ""
    if node.returns:
        ret = f" -> {_annotation_str(node.returns)}"

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({params}){ret}"


def _docstring_first_line(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    """Extract only the first line of a docstring (or empty string)."""
    try:
        first = ast.get_docstring(node)
        if first:
            return first.strip().splitlines()[0]
    except Exception:
        pass
    return ""


# ── File parser ───────────────────────────────────────────────────────────────

def parse_file(path: pathlib.Path) -> list[dict]:
    """
    Parse a single Python file and return a list of entry dicts:
        {type, name, signature, doc, line, decorators, methods}
    """
    try:
        source = path.read_text(encoding="utf-8")
    except Exception as e:
        return [{"type": "error", "name": str(path), "error": str(e)}]

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        return [{"type": "syntax_error", "name": str(path), "error": str(e)}]

    entries = []
    module_imports = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module_imports.append(node)

    # Top-level only (depth=1 inside module)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = [ast.unparse(d) for d in node.decorator_list]
            entries.append({
                "type": "function",
                "name": node.name,
                "signature": _func_signature(node),
                "doc": _docstring_first_line(node),
                "line": node.lineno,
                "decorators": decorators,
            })

        elif isinstance(node, ast.ClassDef):
            bases = [ast.unparse(b) for b in node.bases]
            methods = []
            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    m_decorators = [ast.unparse(d) for d in item.decorator_list]
                    methods.append({
                        "type": "method",
                        "name": item.name,
                        "signature": _func_signature(item),
                        "doc": _docstring_first_line(item),
                        "line": item.lineno,
                        "decorators": m_decorators,
                    })
            entries.append({
                "type": "class",
                "name": node.name,
                "bases": bases,
                "doc": _docstring_first_line(node),
                "line": node.lineno,
                "methods": methods,
            })

    return entries


# ── Directory scanner ─────────────────────────────────────────────────────────

def scan_directory(directory: pathlib.Path, recursive: bool = True) -> dict[pathlib.Path, list[dict]]:
    """Scan a directory and return {file_path: entries} mapping."""
    result = {}
    if not directory.exists():
        return result

    pattern = "**/*.py" if recursive else "*.py"
    for py_file in sorted(directory.glob(pattern)):
        # Skip hidden / venv / cache directories
        if any(part in SKIP_DIRS for part in py_file.parts):
            continue
        entries = parse_file(py_file)
        if entries:  # skip empty files
            result[py_file] = entries
    return result


# ── Formatters ────────────────────────────────────────────────────────────────

INDENT = "    "
LINE_WIDTH = 88


def _wrap_signature(sig: str, indent: str) -> str:
    """Wrap long signatures gracefully."""
    if len(indent + sig) <= LINE_WIDTH:
        return indent + sig
    # Try wrapping at commas
    try:
        paren_open = sig.index("(")
        prefix = sig[:paren_open + 1]
        suffix_start = sig.rindex(")")
        params_str = sig[paren_open + 1:suffix_start]
        suffix = sig[suffix_start:]
        params = [p.strip() for p in params_str.split(",")]
        inner_indent = indent + " " * (paren_open + 1)
        joined = (",\n" + inner_indent).join(params)
        return f"{indent}{prefix}\n{inner_indent}{joined}\n{indent}{suffix}"
    except ValueError:
        return indent + sig


def format_text(file_map: dict[pathlib.Path, list[dict]], root: pathlib.Path) -> str:
    """Render the codebase map as readable text."""
    lines = []
    lines.append("=" * LINE_WIDTH)
    lines.append("  CODEBASE MAP  (AST-extracted signatures)")
    lines.append("=" * LINE_WIDTH)

    for file_path, entries in file_map.items():
        rel = file_path.relative_to(root)
        lines.append("")
        lines.append(f"{'─' * LINE_WIDTH}")
        lines.append(f"  {rel}  (line counts: {file_path.stat().st_size // 1024} KB)")
        lines.append(f"{'─' * LINE_WIDTH}")

        if not entries:
            lines.append("  (no top-level functions or classes)")
            continue

        for entry in entries:
            if entry["type"] == "error":
                lines.append(f"  [READ ERROR] {entry.get('error', '')}")
                continue
            if entry["type"] == "syntax_error":
                lines.append(f"  [SYNTAX ERROR] {entry.get('error', '')}")
                continue

            lines.append("")
            if entry["type"] == "function":
                for dec in entry.get("decorators", []):
                    lines.append(f"  @{dec}")
                lines.append(_wrap_signature(entry["signature"], "  ") + f"   # L{entry['line']}")
                if entry["doc"]:
                    lines.append(f'    """{entry["doc"]}"""')

            elif entry["type"] == "class":
                bases_str = f"({', '.join(entry['bases'])})" if entry.get("bases") else ""
                lines.append(f"  class {entry['name']}{bases_str}:   # L{entry['line']}")
                if entry["doc"]:
                    lines.append(f'    """{entry["doc"]}"""')
                for method in entry.get("methods", []):
                    for dec in method.get("decorators", []):
                        lines.append(f"    @{dec}")
                    lines.append(_wrap_signature(method["signature"], INDENT) + f"   # L{method['line']}")
                    if method["doc"]:
                        lines.append(f'        """{method["doc"]}"""')

    lines.append("")
    lines.append("=" * LINE_WIDTH)
    return "\n".join(lines)


def format_json(file_map: dict[pathlib.Path, list[dict]], root: pathlib.Path) -> str:
    """Render the codebase map as JSON."""
    out = {}
    for file_path, entries in file_map.items():
        rel = str(file_path.relative_to(root))
        out[rel] = entries
    return json.dumps(out, indent=2, ensure_ascii=False)


# ── Grep mode ─────────────────────────────────────────────────────────────────

def grep_map(file_map: dict[pathlib.Path, list[dict]], query: str, root: pathlib.Path) -> str:
    """Find all functions/methods/classes whose name contains `query`."""
    query_lower = query.lower()
    lines = [f"Search: '{query}' in signatures\n{'─' * 60}"]
    found = 0

    for file_path, entries in file_map.items():
        rel = file_path.relative_to(root)
        for entry in entries:
            if entry.get("type") in ("error", "syntax_error"):
                continue
            if query_lower in entry["name"].lower():
                lines.append(f"\n  {rel}  L{entry['line']}")
                if entry["type"] == "function":
                    lines.append(f"    {entry['signature']}")
                else:
                    lines.append(f"    class {entry['name']}")
                if entry.get("doc"):
                    lines.append(f'    # {entry["doc"]}')
                found += 1
            # also search inside class methods
            for method in entry.get("methods", []):
                if query_lower in method["name"].lower():
                    lines.append(f"\n  {rel}  L{method['line']}")
                    lines.append(f"    [{entry['name']}] {method['signature']}")
                    if method.get("doc"):
                        lines.append(f"    # {method['doc']}")
                    found += 1

    lines.append(f"\n{'─' * 60}")
    lines.append(f"Found: {found} match(es)")
    return "\n".join(lines)


# ── Summary stats ─────────────────────────────────────────────────────────────

def summary_stats(file_map: dict[pathlib.Path, list[dict]]) -> str:
    """Print a quick statistics summary at the end."""
    n_files = len(file_map)
    n_funcs = sum(1 for entries in file_map.values() for e in entries if e.get("type") == "function")
    n_classes = sum(1 for entries in file_map.values() for e in entries if e.get("type") == "class")
    n_methods = sum(
        len(e.get("methods", []))
        for entries in file_map.values()
        for e in entries
        if e.get("type") == "class"
    )
    return (
        f"\n  Files: {n_files} | "
        f"Functions: {n_funcs} | "
        f"Classes: {n_classes} | "
        f"Methods: {n_methods}"
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

def build_file_map(target_dirs: list[str]) -> dict[pathlib.Path, list[dict]]:
    """Scan target directories and merge into a single ordered dict."""
    file_map: dict[pathlib.Path, list[dict]] = {}
    for d in target_dirs:
        # "." means root-level files only (non-recursive for root)
        if d == ".":
            for py_file in sorted(ROOT.glob("*.py")):
                if any(part in SKIP_DIRS for part in py_file.parts):
                    continue
                entries = parse_file(py_file)
                if entries:
                    file_map[py_file] = entries
        else:
            directory = ROOT / d
            file_map.update(scan_directory(directory))
    return file_map


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="AST codebase map — extracts function/class signatures from Python files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dirs",
        nargs="*",
        default=DEFAULT_DIRS,
        metavar="DIR",
        help="Directories to scan (default: src approximated math_statistics tests .)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of text",
    )
    parser.add_argument(
        "--grep",
        metavar="NAME",
        help="Search for a function/class name (case-insensitive substring)",
    )
    parser.add_argument(
        "--out",
        metavar="FILE",
        help="Write output to a file instead of stdout",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Suppress summary statistics",
    )

    args = parser.parse_args()

    file_map = build_file_map(args.dirs)

    if args.grep:
        output = grep_map(file_map, args.grep, ROOT)
    elif args.json:
        output = format_json(file_map, ROOT)
    else:
        output = format_text(file_map, ROOT)
        if not args.no_stats:
            output += summary_stats(file_map)

    if args.out:
        pathlib.Path(args.out).write_text(output, encoding="utf-8")
        print(f"[OK] Written to {args.out}  ({len(output):,} chars)")
    else:
        print(output)


if __name__ == "__main__":
    main()
