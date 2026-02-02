"""
Diagnose the exact syntax error in main.py
"""

MAIN_PY = "backend/app/main.py"

print("="*80)
print("DIAGNOSING SYNTAX ERROR")
print("="*80)

with open(MAIN_PY, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# Show lines around the error (line 970)
error_line = 970
start = max(0, error_line - 20)
end = min(len(lines), error_line + 10)

print(f"\nShowing lines {start}-{end} (error at line {error_line}):")
print("="*80)

for i in range(start, end):
    line_num = i + 1
    marker = " >>> ERROR <<<" if line_num == error_line else ""
    print(f"{line_num:4d}: {lines[i].rstrip()}{marker}")

print("="*80)

# Search for all function definitions around this area
print("\nSearching for function definitions near line 970...")
for i in range(max(0, error_line - 30), min(len(lines), error_line + 5)):
    line = lines[i]
    if 'def ' in line or 'async def' in line or '@app.' in line:
        print(f"Line {i+1}: {line.rstrip()}")