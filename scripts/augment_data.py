import os
import pandas as pd
import time
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# 1. 환경 설정
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# OpenRouter 클라이언트 설정
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    default_headers={
        "HTTP-Referer": "none",
        "X-Title": "Clean-Lingual Data Augmentation",
    }
)

# 경로 설정
RAW_FILE = os.path.join("data", "processed", "malicious_raw.tsv")
SEED_FILE = os.path.join("data", "processed", "seed_data.tsv")
OUTPUT_FILE = os.path.join("data", "processed", "malicious_purified_test.tsv")

# 테스트해볼 무료 모델 리스트 (한도 초과 시 교체용)
MODELS = [
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "google/gemma-3n-e2b-it:free",
    "tngtech/deepseek-r1t2-chimera:free",
    "nousresearch/hermes-3-llama-3.1-405b:free"
]

def load_context():
    seed_df = pd.read_csv(SEED_FILE, sep="\t")

    # 3개만 무작위 추출 (데이터가 적을 경우를 대비해 min 처리)
    sample_size = min(3, len(seed_df))
    subset = seed_df.sample(sample_size)

    context = "다음은 원래의 거친 속마음을 순화한 문장 예시이다:\n"
    for _, row in subset.iterrows():
        context += f"{row['source']}\t{row['target']}\n"
    return context

def augment_data(limit=None):
    if not api_key:
        print("[Error] OPENROUTER_API_KEY가 필요하다.")
        return

    raw_df = pd.read_csv(RAW_FILE, sep="\t")
    if limit:
        raw_df = raw_df.head(limit)
    
    if os.path.exists(OUTPUT_FILE):
        processed_df = pd.read_csv(OUTPUT_FILE, sep="\t")
        processed_texts = set(processed_df["source"].tolist())
        print(f"이미 처리된 {len(processed_texts)}개 문장을 건너뛰고 시작한다.")
    else:
        processed_texts = set()

    few_shot_context = load_context()
    system_instruction = f"""
너는 감정적인 문장을 이성적이고 정중한 언어로 다듬는 '비즈니스 소통 중재자'다. 
입력된 문장은 사용자의 '정제되지 않은 감정적 속마음'이다. 너는 이를 공식적인 사회생활에서 사용할 수 있는 '가장 표준적이고 예의 바른 대화체'로 변환하라.

가공 규칙:
1. **일상적 정중함**: 해요체나 하십시오체로 작성하라.
2. **비꼬기 금지**: 과도한 비유로 비꼬는 인상을 주지 마라.
3. **담백한 문장**: 핵심 비판 내용만 담백하고 예의 바르게 전달하라.
4. **의도 보존**: 화자가 무엇에 대해 불만을 가졌는지는 명확히 살려라.
5. **출력 제약**: 부연 설명 없이 오직 [순화된 문장]만 출력하라.

{few_shot_context}
"""

    print(f"총 {len(raw_df)}개 중 남은 작업을 시작한다...")

    current_model_idx = 0

    for i, row in tqdm(raw_df.iterrows(), total=len(raw_df)):
        text = row["문장"]
        if text in processed_texts: continue
            
        success = False
        retry_count = 0
        
        while not success and retry_count < len(MODELS):
            model_name = MODELS[current_model_idx]
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": text}
                    ],
                    temperature=0.3,
                    extra_body={"reasoning": {"enabled": False}}
                )
                
                purified_text = response.choices[0].message.content.strip() if response.choices else "SKIP"
                purified_text = purified_text.replace("[순화된 문장]", "").replace("[", "").replace("]", "").strip()

                new_row = pd.DataFrame({"source": [text], "target": [purified_text]})
                new_row.to_csv(OUTPUT_FILE, mode='a', header=not os.path.exists(OUTPUT_FILE), 
                               index=False, sep="\t", encoding="utf-8-sig")
                
                success = True
                time.sleep(1) # 휴식

            except Exception as e:
                if "429" in str(e):
                    print(f"\n[Quota Error] {model_name} 한도 초과. 다음 모델로 교체한다...")
                    current_model_idx = (current_model_idx + 1) % len(MODELS)
                    retry_count += 1
                    time.sleep(2)
                else:
                    print(f"\n[Error] {e}")
                    break # 다른 에러면 다음 문장으로

    print(f"\n작업 완료! 저장 위치: {OUTPUT_FILE}")

if __name__ == "__main__":
    augment_data(limit=20)
