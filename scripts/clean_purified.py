
import os

purified_file = r'd:\Dev\clean-lingual\data\processed\malicious_purified.tsv'

with open(purified_file, 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()

# Keep lines 1-377 (index 0 to 376)
# Drop lines 378-474 (index 377 to 473)
# Keep lines 475-574 (index 474 to end)

cleaned_lines = lines[:377] + lines[474:]

with open(purified_file, 'w', encoding='utf-8-sig') as f:
    f.writelines(cleaned_lines)

print(f"Cleaned file. Total lines now: {len(cleaned_lines)}")
