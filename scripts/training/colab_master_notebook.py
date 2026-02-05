# %% [markdown]
# # ======================================================================================
# # Clean-Lingual: Stage 1 Classifier Training Notebook (GitHub Data v0.5)
# # ======================================================================================
# 이 파일은 GitHub에서 데이터를 직접 불러와 학습하는 코랩용 마스터 스크립트입니다.

# %% [CELL 1] 필수 라이브러리 설치 및 환경 설정
!pip install -q transformers[torch] datasets evaluate scikit-learn

import os
import pandas as pd
import torch
import numpy as np
import random
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
import evaluate

# 가상환경 및 로깅 관련 설정
os.environ["WANDB_DISABLED"] = "true" 

# %% [CELL 2] GitHub 데이터 동기화
# 저장소 이름 정의
REPO_NAME = "clean-lingual"
REPO_URL = f"https://github.com/yangeval/clean-lingual.git"

# 1. 저장소가 없으면 clone, 있으면 최신 데이터 pull
if not os.path.exists(REPO_NAME):
    print(f"[*] 저장소 클론 중: {REPO_URL}")
    !git clone {REPO_URL}
else:
    print("[*] 기존 저장소 발견. 최신 데이터를 가져옵니다.")
    %cd {REPO_NAME}
    !git pull origin main
    %cd ..

# 2. 데이터 경로 설정 (GitHub 저장소 내의 분할 데이터 위치)
DATA_PATH = os.path.join(REPO_NAME, "data/processed/split/")

# 3. 파일 존재 여부 최종 확인
required_files = ["train.tsv", "val.tsv", "test.tsv"]
all_clear = True
for f in required_files:
    if not os.path.exists(os.path.join(DATA_PATH, f)):
        print(f"[!] 에러: {f} 파일을 찾을 수 없습니다.")
        all_clear = False

if all_clear:
    print(f"[*] 모든 학습 데이터가 준비되었습니다: {DATA_PATH}")

# %% [CELL 3] 데이터셋 전처리 (HuggingFace Format)
MODEL_NAME = "beomi/KcELECTRA-base-v2022"

# 데이터 로드
train_df = pd.read_csv(os.path.join(DATA_PATH, "train.tsv"), sep="\t")
valid_df = pd.read_csv(os.path.join(DATA_PATH, "val.tsv"), sep="\t")
test_df = pd.read_csv(os.path.join(DATA_PATH, "test.tsv"), sep="\t")

def prepare_ds(df):
    return Dataset.from_dict({
        "text": df["source"].astype(str).tolist(),
        "label": df["action"].astype(int).tolist()
    })

dataset = DatasetDict({
    "train": prepare_ds(train_df),
    "valid": prepare_ds(valid_df),
    "test": prepare_ds(test_df)
})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 평가 지표 설정
metric = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="weighted")

# %% [CELL 4] 모델 학습 실행 (Fine-tuning)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch", 
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    compute_metrics=compute_metrics,
)

print("\n" + "="*50)
print("🚀 KcELECTRA 분류기 학습 시작 (v0.5)")
print("="*50)
trainer.train()

# %% [CELL 5] 최종 테스트 및 결과 분석
# 테스트 데이터 예측
test_results = trainer.predict(tokenized_datasets["test"])
print("\n[!] 최종 테스트 메트릭:", test_results.metrics)

# 랜덤 사례 확인 (5개)
preds = np.argmax(test_results.predictions, axis=-1)
labels = test_results.label_ids
test_texts = dataset["test"]["text"]

print("\n" + "="*50)
print("🛡️ 실제 판례 분석 (랜덤 5선)")
print("="*50)
for i in random.sample(range(len(preds)), 5):
    status = "✅ 정답" if labels[i] == preds[i] else "❌ 오답"
    print(f"[{status}] 문장: {test_texts[i]}")
    print(f"      (실제: {labels[i]} / 예측: {preds[i]})\n")

# %% [CELL 6] 모델 내보내기 (Export & Download)
# 모델 저장
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")

# 압축 및 다운로드
!zip -r final_model_v0.5.zip ./final_model

from google.colab import files
files.download("final_model_v0.5.zip")
