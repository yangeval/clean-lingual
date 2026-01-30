
import csv
import sys
import io

# Force stdout to use utf-8 for safe printing in various terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

raw_path = 'data/raw/unsmile_train.tsv'
processed_path = 'data/processed/clean_lingual_v0.5.tsv'

try:
    # Use utf-8-sig to handle BOM automatically
    with open(raw_path, 'r', encoding='utf-8-sig') as f:
        raw = [row[0] for row in csv.reader(f, delimiter='\t')]

    with open(processed_path, 'r', encoding='utf-8-sig') as f:
        processed = [row[0] for row in csv.reader(f, delimiter='\t')]

    # Ultra sanitize for robust comparison
    import re
    def sanitize(text):
        # Keep only Korean, English, and Numbers to ignore punctuation, emojis, and repeated characters like 'ㅋ'
        t = re.sub(r'[^가-힣a-zA-Z0-0]', '', text)
        return t

    raw_clean = [sanitize(r) for r in raw]
    processed_clean = [sanitize(p) for p in processed]

    # Align header
    if len(raw_clean) > 0:
        raw_clean[0] = "source"

    match_found = False
    for i in range(len(processed_clean)):
        if raw_clean[i] != processed_clean[i]:
            print(f"\n[!] Discrepancy found at Line ID (Editor Line): {i + 1}")
            print(f"Expected (Raw Original): '{raw[i]}'")
            print(f"Actual (Processed Original): '{processed[i]}'")
            
            # Look for missing rows
            found_ahead = False
            for look_ahead in range(1, 5):
                if i + look_ahead < len(raw_clean) and raw_clean[i + look_ahead] == processed_clean[i]:
                    print(f"\n[Missing Row Found] {look_ahead} row(s) MISSING before Line {i + 1}.")
                    for k in range(look_ahead):
                        print(f"-> Missing: {raw[i+k]}")
                    found_ahead = True
                    break
            
            if not found_ahead:
                # If not a simple skip, check if current processed matches a later raw entry anyway
                match_found = True # Still count as discrepancy to investigate
                continue # Keep looking for a real skip
            
            match_found = True
            break
    
    if not match_found:
        if len(raw) > len(processed):
            print(f"\n[OK] All {len(processed)} lines match, but Processed is shorter than Raw.")
            print(f"Next expected line from Raw (ID {len(processed)+1}): {raw[len(processed)]}")
        else:
            print("\n[OK] Both files match perfectly.")

except Exception as e:
    # Still print error with utf-8 if possible
    print(f"Error: {str(e)}")
