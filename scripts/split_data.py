import pandas as pd
import os

# 1. 경로 설정
DATA_DIR = os.path.join("data", "processed")
INPUT_FILE = os.path.join(DATA_DIR, "malicious_purified.tsv")
OUTPUT_DIR = os.path.join("data", "train_data")
def split_data():
    # 저장 디렉토리 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[Info] 디렉토리 생성: {OUTPUT_DIR}")

    # 데이터 로드
    if not os.path.exists(INPUT_FILE):
        print(f"[Error] 파일을 찾을 수 없습니다: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE, sep="\t")
    print(f"[Info] 총 데이터 개수: {len(df)}")

    # 데이터 무작위 셔플링
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 분할 포인트 계산
    num_total = len(df)
    num_train = int(num_total * 0.8)
    num_valid = int(num_total * 0.1)
    
    # 데이터 분할
    train_df = df.iloc[:num_train]
    valid_df = df.iloc[num_train:num_train + num_valid]
    test_df = df.iloc[num_train + num_valid:]

    # 결과 저장
    split_info = {
        "train": train_df,
        "valid": valid_df,
        "test": test_df
    }

    for name, data in split_info.items():
        # 1. 통합 TSV 저장 (source, target 포함)
        tsv_path = os.path.join(OUTPUT_DIR, f"{name}.tsv")
        data.to_csv(tsv_path, sep="\t", index=False, encoding="utf-8-sig")
        
        # 2. OpenNMT 학습을 위한 별도 파일 저장 (src, tgt)
        src_path = os.path.join(OUTPUT_DIR, f"{name}.src")
        tgt_path = os.path.join(OUTPUT_DIR, f"{name}.tgt")
        
        data["source"].to_csv(src_path, index=False, header=False, encoding="utf-8")
        data["target"].to_csv(tgt_path, index=False, header=False, encoding="utf-8")

        print(f"[Success] {name} 세트 저장 완료: {len(data)}개")

if __name__ == "__main__":
    try:
        split_data()
    except ImportError:
        print("[Error] scikit-learn 이 설치되어 있지 않습니다. 'pip install scikit-learn'을 실행해 주세요.")
    except Exception as e:
        print(f"[Error] {e}")
