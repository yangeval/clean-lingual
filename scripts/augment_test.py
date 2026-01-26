import os
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 1. 환경 변수 및 클라이언트 설정
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("[Error] .env 파일에 GEMINI_API_KEY가 설정되지 않았다.")
    exit()

client = genai.Client(api_key=api_key)

def run_sample_test():
    """
    추출된 악플 데이터 중 5개를 무작위로 뽑아 
    Gemini API를 통한 순화 테스트를 진행한다.
    """
    raw_file = os.path.join("data", "processed", "malicious_raw.tsv")
    seed_file = os.path.join("data", "processed", "seed_data.tsv")
    
    if not os.path.exists(raw_file) or not os.path.exists(seed_file):
        print("[Error] 필요한 데이터 파일이 존재하지 않는다.")
        return

    # 데이터 로드
    malicious_df = pd.read_csv(raw_file, sep="\t")
    seed_df = pd.read_csv(seed_file, sep="\t")
    
    # 샘플 5개 무작위 추출
    samples = malicious_df.sample(5)["문장"].tolist()
    
    # 3. 시스템 프롬프트 구성 (프롬프트 레시피 반영)
    few_shot_context = "다음은 원래의 거친 속마음을 순화한 문장 예시이다:\n"
    for _, row in seed_df.iterrows():
        few_shot_context += f"원문: {row['source']} -> 순화문: {row['target']}\n"

    system_instruction = f"""
너는 감정적인 문장을 이성적이고 정중한 언어로 다듬는 '비즈니스 소통 중재자'다. 
입력된 문장은 사용자의 '정제되지 않은 감정적 속마음'이다. 너는 이를 공식적인 사회생활에서 사용할 수 있는 '가장 표준적이고 예의 바른 대화체'로 변환하라.

가공 규칙:
1. **일상적 정중함**: "섭섭함", "안타까움", "존중" 등 일상생활에서 흔히 쓰는 어휘를 사용하여 해요체나 하십시오체로 작성하라.
2. **비꼬기 금지**: 과도한 비유나 극적인 표현을 사용하여 상대를 비꼬는 듯한 인상(Sarcasm)을 주지 않도록 주의하라.
3. **담백한 문장**: 문장을 너무 길게 늘이지 말고, 핵심 비판 내용만 담백하고 예의 바르게 전달하라.
4. **의도 보존**: 화자가 무엇에 대해 불만을 가졌는지는 명확히 살려라.
5. **출력 제약**: 부연 설명 없이 오직 [순화된 문장]만 출력하라.

{few_shot_context}
"""

    # 4. 안전 설정 (정확한 카테고리 명칭 사용)
    safety_settings = [
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    ]

    print("Gemini API 샘플 순화 테스트 시작 (5개)...")
    print("-" * 40)

    for i, text in enumerate(samples, 1):
        try:
            # 모델 활용 (설정 추가)
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                config={
                    "system_instruction": system_instruction,
                    "safety_settings": safety_settings
                },
                contents=text
            )
            
            print(f"[{i}/5]")
            print(f"원본: {text}")
            
            # 응답 텍스트 확인 (필터링된 경우 처리)
            if response.text:
                print(f"순화: {response.text.strip()}")
            else:
                print(f"순화 실패: AI가 유해한 콘텐츠로 판단하여 차단함")
                # 차단 원인 상세 출력 (debug용)
                if response.candidates and response.candidates[0].finish_reason:
                    print(f"차단 사유: {response.candidates[0].finish_reason}")
            
            print("-" * 40)
            
        except Exception as e:
            print(f"[Error] {i}번 문장 처리 중 문제 발생: {e}")

if __name__ == "__main__":
    run_sample_test()
