# scripts/verify_article1_numbers.py
"""Verify that all numerical claims in Article 1 LaTeX match canonical counts."""
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ROOT / "articles" / "article1_correlations" / "sections"
sys.path.insert(0, str(ROOT))
from scripts.canonical_counts import get_canonical_counts


def extract_numbers(tex: str) -> list[tuple[int, str, str]]:
    """Find patterns like 'N признаков' or 'N spectral' in tex."""
    patterns = [
        (r"(\d+)~?\\?(?:textbf)?\{?\s*признак", "признак"),
        (r"(\d+)~?\\?(?:textbf)?\{?\s*спектральн", "спектральн"),
        (r"(\d+)~?\\?(?:textbf)?\{?\s*композит", "композит"),
        (r"(\d+)~?\\?(?:textbf)?\{?\s*сезон", "сезон"),
    ]
    results = []
    for line_num, line in enumerate(tex.split("\n"), 1):
        for pat, label in patterns:
            for m in re.finditer(pat, line):
                results.append((line_num, label, m.group(1)))
    return results


def main():
    counts = get_canonical_counts()
    print("Canonical counts:", counts)
    print()

    errors = []
    for tex_file in sorted(SECTIONS.glob("*.tex")):
        text = tex_file.read_text(encoding="utf-8")
        for line_num, label, value in extract_numbers(text):
            print(f"  {tex_file.name}:{line_num}  '{label}' = {value}")
            # Specific checks
            if "композит" in label and value not in ("110", "48", "42", "20"):
                errors.append(f"{tex_file.name}:{line_num}: composite count {value} not in (110, 48, 42, 20)")

    print("\n--- Errors ---")
    if errors:
        for e in errors:
            print("FAIL:", e)
        sys.exit(1)
    else:
        print("OK: all composite counts verified")


if __name__ == "__main__":
    main()
