import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 1. 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "clean_lingual_v0.5.tsv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "train_data", "v0.5")

def prepare_data():
    # 저장 디렉토리 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[*] 디렉토리 생성: {OUTPUT_DIR}")
    
    nmt_dir = os.path.join(OUTPUT_DIR, "nmt")
    if not os.path.exists(nmt_dir):
        os.makedirs(nmt_dir)
        print(f"[*] 디렉토리 생성: {nmt_dir}")

    # 데이터 로드
    if not os.path.exists(INPUT_FILE):
        print(f"[Error] 파일을 찾을 수 없습니다: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE, sep="\t")
    print(f"[*] 원본 데이터 로드 완료: {len(df)}행")

    # 1. 분류기용 데이터 분할 (Action 0, 1, 2 모두 포함)
    # 8:1:1로 분할
    # stratify=df['action'] 을 통해 각 세트의 라벨 비율을 원본과 동일하게 유지함
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['action'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['action'])

    print(f"[*] 분류기용 데이터 분할 완료:")
    print(f"    - Train: {len(train_df)}")
    print(f"    - Valid: {len(val_df)}")
    print(f"    - Test: {len(test_df)}")

    # 분류기용 TSV 저장
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.tsv"), sep="\t", index=False, encoding="utf-8-sig")
    val_df.to_csv(os.path.join(OUTPUT_DIR, "valid.tsv"), sep="\t", index=False, encoding="utf-8-sig")
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.tsv"), sep="\t", index=False, encoding="utf-8-sig")

    # 2. NMT용 데이터 추출 (Action 0과 2만 사용 - Normal과 Purify)
    # Action 1(Block)은 NMT의 대상이 아님
    nmt_train = train_df[train_df['action'] != 1]
    nmt_val = val_df[val_df['action'] != 1]
    nmt_test = test_df[test_df['action'] != 1]

    print(f"[*] NMT용 데이터 추출 완료 (Action 0, 2):")
    print(f"    - Train: {len(nmt_train)}")
    print(f"    - Valid: {len(nmt_val)}")
    print(f"    - Test: {len(nmt_test)}")

    # NMT용 .src, .tgt 저장
    sets = {
        "train": nmt_train,
        "valid": nmt_val,
        "test": nmt_test
    }

    for name, data in sets.items():
        src_path = os.path.join(nmt_dir, f"{name}.src")
        tgt_path = os.path.join(nmt_dir, f"{name}.tgt")
        
        # 텍스트 파일로 저장 (한 줄에 하나의 문장)
        data['source'].to_csv(src_path, index=False, header=False, encoding="utf-8")
        data['target'].to_csv(tgt_path, index=False, header=False, encoding="utf-8")

    print(f"\n[!] 모든 데이터 분할 및 저장 완료: {OUTPUT_DIR}")

if __name__ == "__main__":
    prepare_data()
