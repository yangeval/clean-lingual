
import csv

raw_path = 'd:/Dev/clean-lingual/data/raw/unsmile_train.tsv'
processed_path = 'd:/Dev/clean-lingual/data/processed/clean_lingual_v0.5.tsv'

def get_data_rows(path, start, end):
    rows = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, row in enumerate(reader):
            line_num = i + 1
            if start <= line_num <= end:
                rows.append((line_num, row[0]))
            if line_num > end:
                break
    return rows

raw_rows = get_data_rows(raw_path, 3001, 3400)
proc_rows = get_data_rows(processed_path, 3001, 3400)

print(f"Raw rows: {len(raw_rows)}")
print(f"Processed rows: {len(proc_rows)}")

missing = []
for i in range(len(raw_rows)):
    raw_ln, raw_txt = raw_rows[i]
    if i < len(proc_rows):
        proc_ln, proc_txt = proc_rows[i]
        if raw_txt != proc_txt:
            print(f"Mismatch at Row {proc_ln}:")
            print(f"  Raw Line {raw_ln}: {raw_txt[:50]}")
            print(f"  Processed Line {proc_ln}: {proc_txt[:50]}")
            missing.append(raw_ln)
    else:
        print(f"Missing Raw Line {raw_ln} in processed file.")
        missing.append(raw_ln)

if not missing:
    print("No mismatches found in range 3001-3400.")
else:
    print(f"Found {len(missing)} discrepancies.")
