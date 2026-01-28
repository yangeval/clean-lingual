import os
import pandas as pd
import time
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# 환경 설정 로드
load_dotenv()

class SafetyScreener:
    def __init__(self, prompt_path, model="google/gemini-2.0-flash-exp:free"):
        self.prompt_path = prompt_path
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            timeout=30.0
        )
        self.model = model

    def load_safety_prompt(self):
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def screen_text(self, system_prompt, user_text):
        """문장의 안전성을 판독합니다."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                temperature=0.0 # 판정의 일관성을 위해 0.0 설정
            )
            raw_result = response.choices[0].message.content.strip()
            return self._parse_result(user_text, raw_result)
        except Exception as e:
            print(f"\n[Error] 호출 실패: {e}")
            return None

    def _parse_result(self, source, raw_result):
        """LLM 응답 [판정] | [점수] | [사유]를 파싱하여 딕셔너리로 반환"""
        try:
            parts = [p.strip() for p in raw_result.split('|')]
            return {
                "source": source,
                "decision": parts[0],
                "score": int(parts[1]),
                "reason": parts[2] if len(parts) > 2 else "N/A"
            }
        except Exception:
            # 파싱 실패 시 기본적으로 안전하지 않음으로 간주
            return {"source": source, "decision": "ERROR", "score": 5, "reason": "Parsing Failed"}

    def run(self, input_path, passed_out, blocked_out, start_idx=0, count=50):
        """전체 프로세스 실행"""
        df = pd.read_csv(input_path, sep="\t")
        target_df = df.iloc[start_idx : start_idx + count]
        system_prompt = self.load_safety_prompt()
        
        passed_list = []
        blocked_list = []
        
        print(f"--- {start_idx}번부터 {count}개 데이터 안전성 판독 시작 ---")
        
        for _, row in tqdm(target_df.iterrows(), total=len(target_df)):
            # 기존 데이터셋의 구조(source, target)를 고려
            text = row['source']
            result = self.screen_text(system_prompt, text)
            
            if result:
                if result['decision'] == 'PASS' and result['score'] < 4:
                    passed_list.append({"source": result['source']})
                else:
                    blocked_list.append(result)
            
            time.sleep(0.5)
            
        # 결과 저장
        if passed_list:
            pd.DataFrame(passed_list).to_csv(passed_out, mode='a', sep='\t', index=False, 
                                           header=not os.path.exists(passed_out), encoding='utf-8-sig')
        if blocked_list:
            pd.DataFrame(blocked_list).to_csv(blocked_out, mode='a', sep='\t', index=False, 
                                            header=not os.path.exists(blocked_out), encoding='utf-8-sig')
            
        print(f"\n--- 판독 완료! ---")
        print(f"PASS: {len(passed_list)}건 -> {passed_out}")
        print(f"BLOCK: {len(blocked_list)}건 -> {blocked_out}")

if __name__ == "__main__":
    SCREEN_PROMPT = "prompts/safety_screening_v1.0.md"
    INPUT_DATA = "data/processed/malicious_purified.tsv"
    PASSED_FILE = "data/processed/safety_passed_v1.tsv"
    BLOCKED_FILE = "data/processed/safety_blocked_v1.tsv"
    
    screener = SafetyScreener(SCREEN_PROMPT)
    # 51번부터 50개 테스트 (실제 적용 시 index 조정)
    # screener.run(INPUT_DATA, PASSED_FILE, BLOCKED_FILE, start_idx=50, count=50)
