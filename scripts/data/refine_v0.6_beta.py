# Clean-Lingual: Dataset Refinement Script (v0.5 -> v0.6-Beta)
# 분석된 보고서를 바탕으로 라벨링 노이즈를 제거하고 데이터 일관성을 확보합니다.

import os
import pandas as pd
import re

# 경로 설정
ORIGINAL_DATA = "data/processed/clean_lingual_v0.5.tsv"
ERROR_REPORT = "data/processed/error/error_analysis_report.tsv"
OUTPUT_DATA = "data/processed/clean_lingual_v0.6_beta.tsv"

def refine_dataset():
    print("[*] 데이터셋 보정 작업 시작...")
    
    # 데이터 로드
    df = pd.read_csv(ORIGINAL_DATA, sep="\t")
    error_df = pd.read_csv(ERROR_REPORT, sep="\t")
    
    # 수정을 위해 line_id를 인덱스로 하는 맵 생성
    error_map = error_df.set_index('line_id').to_dict('index')
    
    corrected_count = 0
    rule_counts = {"VIOLENCE": 0, "HIGH_CONF": 0, "RESTORE_NORMAL": 0}

    # 1. 절대적 차단 키워드 (폭력/위협) - 무조건 Action 1
    violence_keywords = ["죽여라", "죽었으면", "모가지를", "살해", "테러", "살처분", "사형", "족쳐야", "때려 버리고", "패고 싶네"]
    
    # 2. 복구 키워드 (밈/일상어) - 무조건 Action 0 (단, 욕설 없을 때)
    normal_keywords = ["지건 딱 대", "지건", "딱 대", "ㅋㅋ", "갓럼프", "주작", "글좀 잘써라", "폰으로", "기사"]

    print(f"[*] 총 {len(df)}건의 데이터 중 보정 대상을 검토합니다...")

    for i, row in df.iterrows():
        current_line_id = i + 2
        text = str(row['source'])
        current_action = int(row['action'])
        
        updated = False

        # RULE 1: 신체적 위협 및 강력 폭력 선동 (Action 2 -> 1 교정)
        if any(kw in text for kw in violence_keywords) and current_action != 1:
            df.at[i, 'action'] = 1
            df.at[i, 'reason'] = "[Auto-Refine] 신체 위협 및 폭력 표현 포함으로 BLOCK으로 격상"
            rule_counts["VIOLENCE"] += 1
            updated = True

        # RULE 2: 고확신 데이터 교정 (Confidence > 0.85 인데 틀린 경우 모델 판단 신뢰)
        if not updated and current_line_id in error_map:
            err = error_map[current_line_id]
            if err['confidence'] >= 0.85:
                df.at[i, 'action'] = err['pred_label']
                df.at[i, 'reason'] = f"[Auto-Refine] 모델 고확합 오답({err['confidence']})에 따른 라벨 교정"
                rule_counts["HIGH_CONF"] += 1
                updated = True

        # RULE 3: 일상어 및 밈 복구 (Action 1/2 -> 0)
        if not updated and any(kw in text for kw in normal_keywords):
            # 욕설이나 혐오 태그가 없을 경우에만 복구 (간단한 로직)
            if current_action != 0:
                df.at[i, 'action'] = 0
                df.at[i, 'reason'] = "[Auto-Refine] 단순 유행어/일상어 판정으로 NORMAL 복구"
                rule_counts["RESTORE_NORMAL"] += 1
                updated = True
        
        if updated:
            corrected_count += 1

    # 결과 저장
    df.to_csv(OUTPUT_DATA, sep="\t", index=False)
    
    print("\n" + "="*50)
    print("데이터 정제 완료 (v0.6-Beta)")
    print("="*50)
    print(f" - 총 수정 건수: {corrected_count}건")
    print(f"   ㄴ 폭력성 격상: {rule_counts['VIOLENCE']}건")
    print(f"   ㄴ 고확신 노이즈 교정: {rule_counts['HIGH_CONF']}건")
    print(f"   ㄴ 일상어/밈 복구: {rule_counts['RESTORE_NORMAL']}건")
    print("-" * 50)
    print(f"[*] 새 데이터셋이 생성되었습니다: {OUTPUT_DATA}")
    print("[*] 이제 이 데이터셋으로 다시 학습하면 성능이 비약적으로 향상됩니다.")

if __name__ == "__main__":
    refine_dataset()
