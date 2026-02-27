import os
from pathlib import Path

def sanitize_file(path_str):
    p = Path(path_str)
    try:
        content = p.read_text(encoding="utf-8")
        
        # Replace common unicode chars causing issues on Windows cp1252
        replacements = {
            "──": "==",
            "✓": "OK",
            "⚠": "WARN",
            "✗": "X",
            "─": "-"
        }
        
        changed = False
        for k, v in replacements.items():
            if k in content:
                content = content.replace(k, v)
                changed = True
                
        if changed:
            p.write_text(content, encoding="utf-8")
            print(f"Sanitized: {p.name}")
            
    except Exception as e:
        print(f"Failed to process {p.name}: {e}")

src_dir = Path("c:/Claude/science_article/src")
bat_dir = Path("c:/Claude/science_article/bat")

# Process python files
for root, _, files in os.walk(src_dir):
    for f in files:
        if f.endswith(".py"):
            sanitize_file(os.path.join(root, f))

# Process bat files (just in case they have unicode echoes)
for root, _, files in os.walk(bat_dir):
    for f in files:
        if f.endswith(".bat"):
            sanitize_file(os.path.join(root, f))

print("Sanitization complete.")
