# Clean-Lingual: Dataset Audit & Consistency Check Script
# 이 스크립트는 학습된 모델을 사용하여 전체 데이터셋의 라벨 일관성을 진단합니다.

import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. 설정 (필요에 따라 경로를 수정하세요)
DATA_PATH = "data/processed/clean_lingual_v0.5.tsv"
MODEL_PATH = "models/classifier/classifier_v0.5"  # 로컬에 다운로드한 모델 폴더 경로
REPORT_PATH = "data/processed/error_analysis_report.tsv"

def run_audit():
    print(f"[*] 진단 프로세스 시작...")
    
    # 모델 및 토크나이저 로드
    if not os.path.exists(MODEL_PATH):
        print(f"[!] 에러: 모델 폴더를 찾을 수 없습니다: {MODEL_PATH}")
        print("[!] 코랩에서 다운로드한 'final_model'의 압축을 해당 경로에 풀어주세요.")
        return

    print(f"[*] 모델 로드 중: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    # GPU 가속 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[*] 사용 디바이스: {device}")

    # 데이터셋 로드
    if not os.path.exists(DATA_PATH):
        print(f"[!] 에러: 데이터셋 파일을 찾을 수 없습니다: {DATA_PATH}")
        return
        
    df = pd.read_csv(DATA_PATH, sep="\t")
    print(f"[*] 데이터셋 로드 완료: {len(df)}건")

    # 추론 시작
    results = []
    batch_size = 16
    
    print(f"[*] 전수 조사 진행 중...")
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i+batch_size].copy()
        texts = batch_df["source"].astype(str).tolist()
        labels = batch_df["action"].astype(int).tolist()
        
        # 토크나이징
        inputs = tokenizer(
            texts, 
            return_tensors="pt", 
            truncation=True, 
            max_length=128, 
            padding="max_length"
        ).to(device)
        
        # 예측
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            confidences, predictions = torch.max(probs, dim=-1)
            
        # 결과 수집
        for idx, (target_idx, text) in enumerate(zip(batch_df.index, texts)):
            true_label = labels[idx]
            pred_label = predictions[idx].item()
            conf = confidences[idx].item()
            
            # 오답 또는 불일치 건만 기록하려면 if true_label != pred_label: 등을 사용
            results.append({
                "line_id": target_idx + 2, # 헤더+1, 0-index+1 고려
                "source": text,
                "true_label": true_label,
                "pred_label": pred_label,
                "is_correct": true_label == pred_label,
                "confidence": round(conf, 4),
                "category": batch_df.iloc[idx].get("category", "N/A"),
                "reason": batch_df.iloc[idx].get("reason", "N/A")
            })

    # 결과 결과물 생성
    full_results_df = pd.DataFrame(results)
    error_df = full_results_df[full_results_df["is_correct"] == False]
    
    # 보고서 저장 (오답 리스트)
    error_df.to_csv(REPORT_PATH, sep="\t", index=False)
    
    # 통계 출력
    print("\n" + "="*60)
    print("📊 데이터셋 정밀 진단 요약 보고서")
    print("="*60)
    print(f" - 전체 검사 행: {len(df)}건")
    print(f" - 일치(Correct): {len(df) - len(error_df)}건")
    print(f" - 불일치(Conflict): {len(error_df)}건")
    print(f" - 데이터 일관성 지수: {((len(df) - len(error_df))/len(df))*100:.2f}%")
    print("-" * 60)
    
    print("\n[!] 카테고리별 불일치 순위 (Top 5):")
    print(error_df["category"].value_counts().head(5))
    
    print("\n[!] 가장 확신에 찬 오답 (Label Noise 의심군):")
    # 모델은 확신하는데 정답 라벨과 다른 경우입니다.
    print(error_df.sort_values(by="confidence", ascending=False)[["line_id", "source", "true_label", "pred_label", "confidence"]].head(10))
    
    print("-" * 60)
    print(f"[*] 상세 오답 리스트가 저장되었습니다: {REPORT_PATH}")
    print("[*] 이 리스트를 참고하여 v0.6 데이터셋의 라벨을 보정하세요.")

if __name__ == "__main__":
    run_audit()
