import csv
import sys

# Windows console encoding handling
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

def check_tsv(file_path):
    print(f"Checking: {file_path}")
    
    inconsistent_clean = []
    block_mismatch = []
    normal_mismatch = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        try:
            header = next(reader)
        except StopIteration:
            return
            
        for i, row in enumerate(reader, start=2):
            if len(row) < 7: continue
            
            source, target, action_str, severity, category, reason, tags = row[0], row[1], row[2], row[3], row[4], row[5], row[6]
            
            try:
                action = int(action_str)
            except:
                continue
                
            # [1] Inconsistent 'clean' tags for toxic actions (Action 1 or 2 with tag 'clean')
            if action != 0 and tags.strip() == 'clean':
                inconsistent_clean.append((i, source, action, tags))
            
            # [2] Block mode mismatch (Action 1 but target is not fixed message)
            if action == 1 and target != '(차단된 문장입니다)':
                block_mismatch.append((i, source, target))
                
            # [3] Normal mode mismatch (Action 0 but source != target)
            if action == 0 and source != target:
                normal_mismatch.append((i, source, target))

    print("\n=== Audit Results ===")
    print(f"[1] Inconsistent 'clean' tags for toxic actions (Action 1 or 2 with tag 'clean'): {len(inconsistent_clean)}")
    for item in inconsistent_clean[:10]:
        print(f" - Line {item[0]}: Action {item[2]}, Tag '{item[3]}' | {item[1][:40]}")

    print(f"\n[2] Block mode mismatch (Action 1 but target is not fixed): {len(block_mismatch)}")
    for item in block_mismatch[:5]:
        print(f" - Line {item[0]}: Target is '{item[2]}' | {item[1][:40]}")

    print(f"\n[3] Normal mode mismatch (Action 0 but source != target): {len(normal_mismatch)}")
    for item in normal_mismatch[:10]:
        print(f" - Line {item[0]}: Source and Target differ | {item[1][:40]}")

if __name__ == "__main__":
    check_tsv('data/processed/unified_3stage_v1_batch1.tsv')
