# Clean-Lingual: Dataset Master Refinement Script (v0.6-Final)
# 2,340건의 오답 전수 조사를 바탕으로 정밀 보정 룰을 적용합니다.

import os
import pandas as pd

# 경로 설정
ORIGINAL_DATA = "data/processed/clean_lingual_v0.5.tsv"
ERROR_REPORT = "data/processed/error/error_analysis_report.tsv"
OUTPUT_DATA = "data/processed/clean_lingual_v0.6_final.tsv"

def master_refine():
    print("[*] 마스터 데이터 정제 작업 시작 (전수 조사 기반)...")
    
    df = pd.read_csv(ORIGINAL_DATA, sep="\t")
    error_df = pd.read_csv(ERROR_REPORT, sep="\t")
    error_map = error_df.set_index('line_id').to_dict('index')
    
    counts = {"RESTORE_NORMAL": 0, "UPGRADE_BLOCK": 0, "FIX_LABEL_NOISE": 0}

    # 1. 강력 차단 키워드 (Action 2 -> 1)
    block_hard_keywords = [
        "재기해", "모가지", "따버려", "참수", "학살", "살해", "가스실", "몰살", "찢어", "직이뿌", "죽여버",
        "죽여라", "죽어라", "죽어라", "자살해", "칼푹찍", "불질러", "살충", "사형", "참살", "난도질", "거세"
    ]
    
    # 2. 강제 정상 복구 키워드 (Action 2 -> 0)
    normal_hard_keywords = [
        "응원합니다", "감사합니다", "멋지다", "궁금하네", "재밌네", "어디임", "잘봤다", "행복하시길",
        "고생한다", "부럽다", "화이팅", "뉴스", "기사", "멋진", "좋은글", "팡파레"
    ]

    for i, row in df.iterrows():
        line_id = i + 2
        text = str(row['source'])
        current_action = int(row['action'])
        updated = False

        # RULE 1: 신체적 위해/살인 예고 -> 무조건 Action 1 (BLOCK)
        if any(kw in text for kw in block_hard_keywords) and current_action != 1:
            df.at[i, 'action'] = 1
            df.at[i, 'reason'] = f"[Master-Refine] 극단적 폭력 표현({text[:20]}...) BLOCK 격상"
            counts["UPGRADE_BLOCK"] += 1
            updated = True

        # RULE 2: 고확신 모델 판단 수용 (Confidence > 0.9) - 라벨 노이즈 제거
        if not updated and line_id in error_map:
            err = error_map[line_id]
            if err['confidence'] >= 0.90:
                df.at[i, 'action'] = err['pred_label']
                df.at[i, 'reason'] = f"[Master-Refine] 모델 고확신({err['confidence']}) 기반 라벨 수정"
                counts["FIX_LABEL_NOISE"] += 1
                updated = True

        # RULE 3: 문장 구조 및 키워드 기반 정상 복구 -> Action 0 (NORMAL)
        # 키워드가 있거나, 물음표로 끝나면서 확신도가 낮은 경우
        if not updated and current_action != 0:
            is_normal_sign = any(kw in text for kw in normal_hard_keywords) or text.strip().endswith("?")
            if is_normal_sign:
                # 단, 비속어가 직접적으로 포함되지 않았을 때만 구제 (아주 기초적 검사)
                bad_signs = ["시발", "병신", "존나", "새끼", "련"]
                if not any(bs in text for bs in bad_signs):
                    df.at[i, 'action'] = 0
                    df.at[i, 'reason'] = "[Master-Refine] 일상적 맥락/질문형 문장 NORMAL 복구"
                    counts["RESTORE_NORMAL"] += 1
                    updated = True

    df.to_csv(OUTPUT_DATA, sep="\t", index=False)
    
    print("\n" + "="*50)
    print("마스터 데이터 정제 완료 (v0.6-Final)")
    print("="*50)
    print(f" - 일상어/질문 복구: {counts['RESTORE_NORMAL']}건")
    print(f" - 폭력성 BLOCK 격상: {counts['UPGRADE_BLOCK']}건")
    print(f" - 라벨 노이즈 교정: {counts['FIX_LABEL_NOISE']}건")
    print("-" * 50)
    print(f"총 {sum(counts.values())}건의 데이터가 수술되었습니다.")
    print(f"결과 파일: {OUTPUT_DATA}")

if __name__ == "__main__":
    master_refine()
