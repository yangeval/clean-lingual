import os
import pandas as pd
from datasets import load_dataset

def download_data():
    """
    HuggingFace에서 kor_unsmile 데이터셋을 다운로드하고, 
    관리하기 편한 TSV 형식으로 data/raw 폴더에 저장.
    """
    print("UnSmile 데이터셋 다운로드 중 (HuggingFace)...")
    
    # 1. 데이터셋 로드
    try:
        dataset = load_dataset("smilegate-ai/kor_unsmile")
    except Exception as e:
        print(f" 다운로드 실패: {e}")
        return

    # 2. 판다스 데이터프레임으로 변환
    train_df = pd.DataFrame(dataset["train"])
    valid_df = pd.DataFrame(dataset["valid"])

    # 3. 저장 경로 설정 (data/raw)
    raw_path = os.path.join("data", "raw")
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)

    # 4. TSV 파일로 저장 (sep='\t' 설정)
    # 문장 내의 쉼표(,)로부터 안전하게 탭으로 구분.
    train_file = os.path.join(raw_path, "unsmile_train.tsv")
    valid_file = os.path.join(raw_path, "unsmile_valid.tsv")

    # utf-8-sig로 저장(엑셀에서 한글 깨짐 방지)
    train_df.to_csv(train_file, index=False, sep="\t", encoding="utf-8-sig")
    valid_df.to_csv(valid_file, index=False, sep="\t", encoding="utf-8-sig")

    print(" 다운로드 및 변환 완료!")
    print(f"   - 저장 위치 1: {train_file} ({len(train_df)} 문장)")
    print(f"   - 저장 위치 2: {valid_file} ({len(valid_df)} 문장)")

if __name__ == "__main__":
    download_data()