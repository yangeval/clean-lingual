import pandas as pd
import os

def filter_malicious_data():
    """
    UnSmile 데이터셋에서 '악플/욕설' 항목만 필터링하여
    가공하기 편하도록 별도의 파일로 저장.
    """
    raw_file = os.path.join("data", "raw", "unsmile_train.tsv")
    output_dir = os.path.join("data", "processed")
    output_file = os.path.join(output_dir, "malicious_raw.tsv")

    # 1. 파일 존재 여부 확인
    if not os.path.exists(raw_file):
        print(f" 원본 파일을 찾을 수 없습니다: {raw_file}")
        return

    # 2. 데이터 불러오기
    print(f" 데이터를 읽어오는 중: {raw_file}")
    df = pd.read_csv(raw_file, sep="\t")

    # 3. '악플/욕설' 컬럼이 1인 데이터 필터링
    # 참고: 다른 혐오 레이블이 섞여 있을 수 있지만, 일단 '악플/욕설'이 포함된 모든 문장을 가져온다.
    malicious_df = df[df["악플/욕설"] == 1]

    # 4. '문장' 컬럼만 추출 (나중에 순화할 대상)
    # 중복된 문장이 있을 수 있으므로 정비한다.
    result_df = malicious_df[["문장"]].drop_duplicates()

    # 5. 저장 경로 생성 및 저장
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_df.to_csv(output_file, index=False, sep="\t", encoding="utf-8-sig")

    print(f" 필터링 완료!")
    print(f"   - 총 {len(df)}개 문장 중 {len(malicious_df)}개의 악플/욕설 문장 발견")
    print(f"   - 중복 제거 후 {len(result_df)}개 문장 저장 완료")
    print(f"   - 저장 위치: {output_file}")

if __name__ == "__main__":
    filter_malicious_data()
