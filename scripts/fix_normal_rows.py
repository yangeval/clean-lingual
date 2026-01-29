import csv
import os

file_path = 'data/processed/unified_3stage_v1_batch1.tsv'
temp_path = file_path + '.tmp'

with open(file_path, 'r', encoding='utf-8') as fin, open(temp_path, 'w', encoding='utf-8', newline='') as fout:
    reader = csv.reader(fin, delimiter='\t')
    writer = csv.writer(fout, delimiter='\t')
    
    header = next(reader)
    writer.writerow(header)
    
    fixed_count = 0
    for row in reader:
        if len(row) < 7:
            writer.writerow(row)
            continue
            
        action = row[2]
        if action == '0':
            if row[0] != row[1]:
                row[1] = row[0]  # target = source
                fixed_count += 1
        writer.writerow(row)

print(f"Fixed {fixed_count} NORMAL rows where target differed from source.")

os.replace(temp_path, file_path)
