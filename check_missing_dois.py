import re
with open('references.bib', 'r', encoding='utf-8') as f:
    content = f.read()
entries = re.findall(r'@\w+\{([^,]+),([^@]+)\}', content)
missing = [key for key, body in entries if 'doi' not in body.lower()]
print('Missing DOIs:', missing)
