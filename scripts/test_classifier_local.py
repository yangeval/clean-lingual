import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# 1. 모델 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "classifier", "classifier_v0.5")

# 2단계: 모델 및 토크나이저 로드 (로컬 환경)
print(f"[*] 모델 로드 중: {MODEL_PATH}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    print("[!] 모델 로드 완료!")
except Exception as e:
    print(f"[Error] 모델 로드 실패: {e}")
    exit()

# 3단계: 추론 함수 정의
def predict(text):
    # 입력 텍스트 토크나이징
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # 추론 모드 전환 (기울기 계산 비활성화)
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 결과 해석 (Softmax를 통해 확률로 변환)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][prediction].item()
    
    # 결과 라벨 매핑
    labels = {0: "NORMAL (통과)", 1: "BLOCK (차단)", 2: "PURIFY (순화)"}
    
    return labels[prediction], confidence

# 4단계: 대화형 테스트 루프
print("\n" + "="*50)
print("   Clean-Lingual AI 판사 (v0.5) 테스트 세션")
print("   종료하려면 'exit'를 입력하세요.")
print("="*50)

while True:
    user_input = input("\n판독할 문장 입력 > ")
    if user_input.lower() == 'exit':
        break
    
    result, score = predict(user_input)
    print(f"--- 판결 결과: {result} (신뢰도: {score:.2%}) ---")
