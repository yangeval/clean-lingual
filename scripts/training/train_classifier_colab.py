# Clean-Lingual: Stage 1 Classifier Training Script (Colab Version)
# 모델: beomi/KcELECTRA-base-v2022
# 목적: Action 0(Normal), 1(Block), 2(Purify) 3단계 분류 학습

import pandas as pd
import torch
import os
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
import evaluate

# 1. 환경 설정 및 데이터 경로
DATA_PATH = "data/train_data/v0.5/"
OUTPUT_DIR = "./results"
MODEL_SAVE_DIR = "./final_model"
MODEL_NAME = "beomi/KcELECTRA-base-v2022"

# WANDB 비활성화 (로그 기록 생략)
os.environ["WANDB_DISABLED"] = "true"

def train():
    # 2. 데이터 로드
    print("[*] 로컬 데이터 로드 중...")
    try:
        train_df = pd.read_csv(os.path.join(DATA_PATH, "train.tsv"), sep="\t")
        valid_df = pd.read_csv(os.path.join(DATA_PATH, "valid.tsv"), sep="\t")
    except FileNotFoundError:
        print("[Error] 데이터를 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    # 3. 데이터셋 변환 (HuggingFace Format)
    def prepare_ds(df):
        return Dataset.from_dict({
            "text": df["source"].astype(str).tolist(),
            "label": df["action"].astype(int).tolist()
        })

    dataset = DatasetDict({
        "train": prepare_ds(train_df),
        "valid": prepare_ds(valid_df)
    })

    # 4. 토크나이저 및 전처리
    print(f"[*] 토크나이저 로드 중: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 5. 평가 지표 (F1 Score) 설정
    metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels, average="weighted")

    # 6. 모델 로드 (3개 라벨 분류용)
    print(f"[*] 모델 로드 중: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    # 7. 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",  # 최신 라이브러리 규격 (eval_strategy)
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_steps=10,
    )

    # 8. Trainer 초기화 및 학습 실행
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        compute_metrics=compute_metrics,
    )

    print("\n" + "="*50)
    print("🚀 학습 시작 (KcELECTRA Classifier v0.5)")
    print("="*50)
    trainer.train()

    # 9. 모델 최종 저장
    print(f"\n[*] 학습 완료! 모델 저장 중: {MODEL_SAVE_DIR}")
    model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    print("[!] 모든 과정이 완료되었습니다.")

if __name__ == "__main__":
    train()
