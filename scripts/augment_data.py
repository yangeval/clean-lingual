import os
import pandas as pd
import time
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# 1. 환경 설정
load_dotenv()

class PromptManager:
    """프롬프트 생성 및 관리를 담당"""
    def __init__(self, seed_file):
        self.seed_file = seed_file
        self.instruction_template = """너는 감정적인 문장을 이성적이고 정중한 언어로 다듬는 '비즈니스 소통 중재자'다. 
입력된 문장은 사용자의 '정제되지 않은 감정적 속마음'이다. 너는 이를 공식적인 사회생활에서 사용할 수 있는 '가장 표준적이고 예의 바른 대화체'로 변환하라.

가공 규칙:
1. **일상적 정중함**: 해요체나 하십시오체로 작성하라.
2. **비꼬기 금지**: 과도한 비유로 비꼬는 인상을 주지 마라.
3. **담백한 문장**: 핵심 비판 내용만 담백하고 예의 바르게 전달하라.
4. **의도 보존**: 화자가 무엇에 대해 불만을 가졌는지는 명확히 살려라.
5. **출력 제약**: 부연 설명 없이 오직 [순화된 문장]만 출력하라.
"""

    def _load_few_shot_examples(self, n=3):
        if not os.path.exists(self.seed_file):
            return ""
        try:
            seed_df = pd.read_csv(self.seed_file, sep="\t")
            sample_size = min(n, len(seed_df))
            subset = seed_df.sample(sample_size)
            
            context = "\n다음은 원래의 거친 속마음을 순화한 문장 예시이다:\n"
            for _, row in subset.iterrows():
                context += f"{row['source']}\t{row['target']}\n"
            return context
        except Exception as e:
            print(f"[Warning] Seed data 로드 실패: {e}")
            return ""

    def get_system_prompt(self):
        few_shot = self._load_few_shot_examples()
        return f"{self.instruction_template}\n{few_shot}"

class LLMHandler:
    """LLM API 호출 및 모델 로테이션을 담당"""
    def __init__(self, api_key, models):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=30.0,
            default_headers={
                "HTTP-Referer": "none",
                "X-Title": "Clean-Lingual Data Augmentation",
            }
        )
        self.models = models
        self.current_model_idx = 0

    def get_current_model(self):
        return self.models[self.current_model_idx]

    def purify_text(self, system_prompt, user_text):
        retry_count = 0
        while retry_count < len(self.models):
            model_name = self.models[self.current_model_idx]
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text}
                    ],
                    temperature=0.3,
                    extra_body={"reasoning": {"enabled": False}}
                )
                
                content = response.choices[0].message.content.strip() if response.choices else "SKIP"
                content = content.replace("[순화된 문장]", "").replace("[", "").replace("]", "").strip()
                return content

            except Exception as e:
                status_code = getattr(e, 'status_code', None)
                
                # 에러 코드별 구체화된 응답 처리
                error_map = {
                    400: f"Bad Request (잘못된 요청: {model_name}의 파라미터나 시스템 프롬프트 미지원 가능성)",
                    401: f"Unauthorized (인증 실패: API 키가 유효하지 않거나 권한이 없음)",
                    404: f"Not Found (모델 찾기 실패: {model_name}이 해당 제공자에 존재하지 않음)",
                    429: f"Too Many Requests (할당량 초과: {model_name}의 속도 제한 도달)",
                    431: f"Request Header Fields Too Large (헤더 크기 초과)"
                }

                if status_code in error_map:
                    print(f"\n[HTTP {status_code}] {error_map[status_code]}")
                    
                    # 401(인증)이나 404(모델 실종)는 모델 교체로 해결되지 않을 가능성이 높으나, 
                    # 규칙에 따라 다음 모델로 전환 시도 (혹은 필요시 여기서 정지 가능)
                    print(f"다음 모델로 교체를 시도합니다... (현재 모델: {model_name})")
                    self.current_model_idx = (self.current_model_idx + 1) % len(self.models)
                    retry_count += 1
                    time.sleep(2)
                else:
                    # 정의되지 않은 기타 에러
                    print(f"\n[Unhandled Error] {model_name}에서 정의되지 않은 에러 발생: {e}")
                    return "ERROR"
                    
        return "SKIP"

class DataAugmentor:
    """전체 데이터 증강 프로세스를 관리"""
    def __init__(self, raw_file, seed_file, output_file, models):
        self.raw_file = raw_file
        self.output_file = output_file
        self.prompt_manager = PromptManager(seed_file)
        self.llm_handler = LLMHandler(os.getenv("OPENROUTER_API_KEY"), models)

    def _get_processed_texts(self):
        if os.path.exists(self.output_file):
            try:
                df = pd.read_csv(self.output_file, sep="\t")
                return set(df["source"].tolist())
            except Exception:
                return set()
        return set()

    def run(self, limit=None):
        if not os.path.exists(self.raw_file):
            print(f"[Error] {self.raw_file} 파일이 없습니다.")
            return

        raw_df = pd.read_csv(self.raw_file, sep="\t")
        if limit:
            raw_df = raw_df.head(limit)

        processed_texts = self._get_processed_texts()
        system_prompt = self.prompt_manager.get_system_prompt()
        
        start_model = self.llm_handler.get_current_model()
        print(f"작업 시작 모델: {start_model}")
        print(f"총 {len(raw_df)}개 중 남은 작업을 시작한다...")

        pbar = tqdm(raw_df.iterrows(), total=len(raw_df))
        for _, row in pbar:
            text = row["문장"]
            if text in processed_texts:
                continue

            current_model = self.llm_handler.get_current_model()
            pbar.set_description(f"Processing ({current_model.split('/')[-1]})")

            purified = self.llm_handler.purify_text(system_prompt, text)
            
            if purified == "SKIP":
                print(f"\n[Terminating] 모든 가용 모델의 할당량이 초과되었습니다. 작업을 중단합니다.")
                break
            
            if purified != "ERROR":
                new_row = pd.DataFrame({"source": [text], "target": [purified]})
                new_row.to_csv(self.output_file, mode='a', 
                               header=not os.path.exists(self.output_file), 
                               index=False, sep="\t", encoding="utf-8-sig")
                # 성공 시에만 텍스트셋에 추가 (중복 처리 방지)
                processed_texts.add(text)
            
            time.sleep(1)

        print(f"\n작업 완료! 저장 위치: {self.output_file}")

if __name__ == "__main__":
    # 경로 설정
    RAW_PATH = os.path.join("data", "processed", "malicious_raw.tsv")
    SEED_PATH = os.path.join("data", "processed", "seed_data.tsv")
    OUTPUT_PATH = os.path.join("data", "processed", "malicious_purified.tsv")
    
    MODELS_LIST = [
        "deepseek/deepseek-chat-v3.1",
        "google/gemini-2.0-flash-exp:free",
        "tngtech/deepseek-r1t2-chimera:free",
        "deepseek/deepseek-r1-0528:free",
        "tngtech/deepseek-r1t-chimera:free",
    ]

    augmentor = DataAugmentor(RAW_PATH, SEED_PATH, OUTPUT_PATH, MODELS_LIST)
    augmentor.run()
