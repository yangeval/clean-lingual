"""
범용 배치 처리 스크립트
사용법: python scripts/process_chunk.py <chunk_file> <start_line> <end_line>

이 스크립트는:
1. chunk_file에서 원본 데이터를 읽음
2. 개수 검증 (start_line ~ end_line과 일치 확인)
3. 에이전트가 라벨링 로직을 채움
4. clean_lingual_v0.5.tsv에 자동 저장
"""

import sys
import csv
import json

def parse_origin_tags(raw_label_str):
    """
    '[0, 1, 0, ...]' 형태의 문자열을 한글 태그 텍스트로 변환
    """
    try:
        # JSON 형태로 파싱 시도
        labels = json.loads(raw_label_str)
        mapping = [
            "여성/가족", "남성", "성소수자", "인종/국적", "연령",
            "지역", "종교", "기타 혐오", "악플/욕설", "clean"
        ]
        tags = [mapping[i] for i, val in enumerate(labels) if val == 1]
        return ", ".join(tags) if tags else "none"
    except:
        return raw_label_str

def main():
    if len(sys.argv) != 4:
        print("Usage: python scripts/process_chunk.py <chunk_file> <start_line> <end_line>")
        sys.exit(1)
    
    chunk_file = sys.argv[1]
    start_line = int(sys.argv[2])
    end_line = int(sys.argv[3])
    expected_count = end_line - start_line + 1
    
    # 1. 원본 데이터 읽기
    with open(chunk_file, 'r', encoding='utf-8') as f:
        raw_data = list(csv.reader(f, delimiter='\t'))
    
    # 2. 자동 검증
    actual_count = len(raw_data)
    if actual_count != expected_count:
        raise ValueError(
            f" 데이터 개수 불일치!\n"
            f"   예상: {expected_count}건 (Line {start_line}~{end_line})\n"
            f"   실제: {actual_count}건\n"
            f"   → 데이터 추출 과정을 재확인하세요."
        )
    
    print(f"[OK] Data validation complete: {actual_count} rows")
    
    # 3. 라벨링 (에이전트가 이 부분을 채움)
    labeled_data = []
    
    # ==========================================
    # 에이전트 작업 영역 (여기만 수정)
    # ==========================================
    
    # 튜플 리스트 형식: [(target, action, severity, category, reason), ...]
    # 원본 데이터 순서와 100% 일치해야 함.
    results = [
        # (예시) ("순화어", 2, 2, "GENDER", "이유"),
    ]
    
    # ==========================================
    # 에이전트 작업 영역 끝
    # ==========================================
    
    # 결과 개수 즉시 검증
    if len(results) != actual_count:
        raise ValueError(
            f" 결과 개수 불일치!\n"
            f"   원본 데이터: {actual_count}건\n"
            f"   라벨링 결과: {len(results)}건\n"
            f"   → results 리스트의 개수를 확인하세요."
        )

    # 데이터 병합
    for i, row in enumerate(raw_data):
        source = row[0]
        # 원본 labels 리스트를 한글 텍스트로 자동 변환
        origin_tags = parse_origin_tags(row[-1])
        
        target, action, severity, category, reason = results[i]
        labeled_data.append([source, target, action, severity, category, reason, origin_tags])
    
    # 4. 저장
    output_file = r'd:\Dev\clean-lingual\data\processed\clean_lingual_v0.5.tsv'
    with open(output_file, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(labeled_data)
    
    print(f"[OK] Saved: {len(labeled_data)} rows -> {output_file}")
    
    # 5. 최종 검증
    if len(labeled_data) != expected_count:
        raise ValueError(
            f" 라벨링 개수 불일치!\n"
            f"   예상: {expected_count}건\n"
            f"   실제: {len(labeled_data)}건"
        )
    
    print(f"[SUCCESS] Batch processing complete: {len(labeled_data)}/{expected_count} rows")

if __name__ == "__main__":
    main()
