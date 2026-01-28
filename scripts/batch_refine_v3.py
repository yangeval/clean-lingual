import os
import pandas as pd
import time
import json
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

class BatchRefiner:
    def __init__(self, sys_prompt_path, safety_prompt_path):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self.model = "google/gemini-2.0-flash-exp:free"
        
        with open(sys_prompt_path, 'r', encoding='utf-8') as f:
            self.refine_instruction = f.read()
        with open(safety_prompt_path, 'r', encoding='utf-8') as f:
            self.safety_instruction = f.read()

    def process_batch(self, batch_df, max_retries=3):
        """문장 뭉치를 한 번의 API 호출로 처리합니다. (재시도 로직 포함)"""
        input_texts = ""
        for i, row in batch_df.iterrows():
            input_texts += f"ID:{i} | TEXT:{row['문장']}\n"

        combined_prompt = f"""
{self.safety_instruction}
---
{self.refine_instruction}
---
위 지침에 따라 아래 문장들을 판독하고 순화하라. 
출력 형식: ID:번호 | [판정] | [유해성점수] | [순화결과]
(주의: BLOCK 판정 시 [순화결과]에는 차단 사유를 짧게 기재하라)
"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": combined_prompt},
                        {"role": "user", "content": input_texts}
                    ],
                    temperature=0.2
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Batch Attempt {attempt+1} Error: {e}")
                if "429" in str(e):
                    wait_time = (attempt + 1) * 10
                    print(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    time.sleep(2)
        return None

    def run(self, input_path, output_path, start_idx=0, count=100, batch_size=20):
        df = pd.read_csv(input_path, sep="\t")
        target_df = df.iloc[start_idx : start_idx + count]
        
        # 파일이 이미 존재하면 읽어오기 (이어하기 지원용)
        if os.path.exists(output_path):
            existing_df = pd.read_csv(output_path, sep="\t")
            processed_ids = existing_df['id_info'].apply(lambda x: int(x.split(':')[1])).tolist()
        else:
            processed_ids = []

        print(f"--- {start_idx}번부터 {count}개 데이터 배치 처리 시작 ---")
        
        for i in range(0, len(target_df), batch_size):
            batch = target_df.iloc[i : i + batch_size]
            
            # 이미 처리된 ID가 포함된 배치는 건너뛰기 (간단한 구현)
            first_id = batch.index[0]
            if first_id in processed_ids:
                continue

            raw_response = self.process_batch(batch)
            
            if raw_response:
                batch_results = []
                for line in raw_response.split('\n'):
                    if "|" in line:
                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) >= 4:
                            batch_results.append({
                                "id_info": parts[0],
                                "decision": parts[1],
                                "score": parts[2],
                                "target": parts[3]
                            })
                
                if batch_results:
                    res_df = pd.DataFrame(batch_results)
                    res_df.to_csv(output_path, mode='a', sep='\t', index=False, 
                                   header=not os.path.exists(output_path), encoding='utf-8-sig', lineterminator='\n')
                    print(f"Batch {i//batch_size + 1} saved.")
            
            time.sleep(2)

        print(f"Batch task completed: {output_path}")

if __name__ == "__main__":
    refiner = BatchRefiner("prompts/purification_v2.1.md", "prompts/safety_screening_v1.0.md")
    # 11번 데이터(인덱스 10)부터 90개 처리 (100번까지)
    refiner.run(
        input_path="data/processed/malicious_raw.tsv", 
        output_path="data/processed/malicious_purified_v2.1_batch2.tsv", 
        start_idx=10, 
        count=90, 
        batch_size=20
    )
