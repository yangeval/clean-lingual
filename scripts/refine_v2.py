import os
import pandas as pd
import time
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# 1. 환경 설정
load_dotenv()

class RefinerV2:
    def __init__(self, prompt_path, input_path, output_path):
        self.prompt_path = prompt_path
        self.input_path = input_path
        self.output_path = output_path
        
        # OpenRouter 설정 (사용자님의 기존 설정 계승)
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            timeout=30.0,
            default_headers={
                "HTTP-Referer": "none",
                "X-Title": "Clean-Lingual Dataset Refinement V2",
            }
        )
        self.model = "google/gemini-2.0-flash-exp:free" # 가성비 좋은 모델 사용

    def load_system_prompt(self):
        """v2.0 프롬프트 파일 내용을 읽어옵니다."""
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def refine_text(self, system_prompt, user_text):
        """단일 문장을 프롬프트 규칙에 따라 정제합니다."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                temperature=0.2, # 일관성을 위해 낮은 온도 설정
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"\n[Error] 호출 실패: {e}")
            return None

    def run(self, start_idx=0, count=100):
        """지정된 범위의 데이터를 처리합니다."""
        # 데이터 로드 (기존 정제된 파일에서 source만 가져오거나 raw 파일 사용)
        df = pd.read_csv(self.input_path, sep="\t")
        
        # 처리할 구간 설정
        target_df = df.iloc[start_idx : start_idx + count]
        system_prompt = self.load_system_prompt()
        
        results = []
        print(f"--- {start_idx}번부터 {start_idx + count}번까지 정제 시작 ---")
        
        for _, row in tqdm(target_df.iterrows(), total=len(target_df)):
            source_text = row['source'] 
            # 정제 실행
            v2_target = self.refine_text(system_prompt, source_text)
            
            if v2_target:
                results.append({"source": source_text, "target": v2_target})
            
            # API 할당량 배려를 위한 미세 지연
            time.sleep(0.5)
            
        # 결과 저장 (Append 모드)
        result_df = pd.DataFrame(results)
        file_exists = os.path.exists(self.output_path)
        result_df.to_csv(self.output_path, mode='a', sep='\t', 
                         index=False, header=not file_exists, encoding='utf-8-sig')
        
        print(f"--- 저장 완료: {self.output_path} ---")

if __name__ == "__main__":
    # 경로 설정
    PROMPT_FILE = "prompts/purification_v2.1.md"
    INPUT_FILE = "data/processed/malicious_purified.tsv"
    OUTPUT_FILE = "data/processed/malicious_purified_v2.1.tsv"
    
    refiner = RefinerV2(PROMPT_FILE, INPUT_FILE, OUTPUT_FILE)
    
    # 0번부터 50개 처리 (2번 라인부터 51번 라인까지)
    refiner.run(start_idx=0, count=50)
