import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_dataset():
    # 설정
    input_file = r'd:\Dev\clean-lingual\data\processed\clean_lingual_v0.5.tsv'
    output_dir = r'd:\Dev\clean-lingual\data\processed\split'
    
    # 1. 데이터 로드
    print(f"데이터 로드 중: {input_file}")
    if not os.path.exists(input_file):
        print(f"에러: 파일을 찾을 수 없습니다. ({input_file})")
        return

    df = pd.read_csv(input_file, sep='\t')
    print(f"총 데이터 수: {len(df)}건")

    # 2. 데이터 분할 (Train 80%, Val 10%, Test 10%)
    # 층화 추출(stratify)을 통해 action 라벨 분포를 유지합니다.
    print("데이터 분할 중 (8:1:1)...")
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['action']
    )

    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42, 
        stratify=temp_df['action']
    )

    # 3. 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 4. 파일 저장
    train_path = os.path.join(output_dir, 'train.tsv')
    val_path = os.path.join(output_dir, 'val.tsv')
    test_path = os.path.join(output_dir, 'test.tsv')

    train_df.to_csv(train_path, sep='\t', index=False)
    val_df.to_csv(val_path, sep='\t', index=False)
    test_df.to_csv(test_path, sep='\t', index=False)

    # 5. 결과 보고
    print("-" * 50)
    print(f"분할 완료!")
    print(f" - Train: {len(train_df)}건 ({len(train_df)/len(df):.1%}) -> {train_path}")
    print(f" - Val:   {len(val_df)}건 ({len(val_df)/len(df):.1%}) -> {val_path}")
    print(f" - Test:  {len(test_df)}건 ({len(test_df)/len(df):.1%}) -> {test_path}")
    print("-" * 50)
    
    # 라벨 분포 확인
    print("\n라벨(action) 분포 확인:")
    print("Original:\n", df['action'].value_counts(normalize=True).sort_index())
    print("\nTrain:\n", train_df['action'].value_counts(normalize=True).sort_index())

if __name__ == "__main__":
    split_dataset()
