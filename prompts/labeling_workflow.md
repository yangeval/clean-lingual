---
**데이터 라벨링 실행 프롬프트 (v0.6.1)**
사용법: `prompts/labeling_workflow.md {START_LINE} {END_LINE}`

## Config
```yaml
LABELING_RULES: "prompts/labeling_rules.md"
SOURCE_DATA: "data/raw/unsmile_train.tsv"
OUTPUT_DATA: "data/processed/clean_lingual_v0.5.tsv"
BATCH_SIZE: 50
START_LINE: {START_LINE}
END_LINE: {END_LINE}
```

**주의**: 규칙 파일은 세션당 1회만 정독. 데이터 파일은 `wc -l` 및 추출 목적 외에는 직접 읽지 말 것(토큰 절약).

---

### Phase 1: 준비
1. `${LABELING_RULES}` 숙지 (이미 읽었다면 skip)
2. `${SOURCE_DATA}` 상단 3줄로 구조 파악
3. `wc -l ${OUTPUT_DATA}`로 현재 진행도 확인
   - **보고**: "준비 완료. [N]줄까지 작업됨 -> Line ${START_LINE}부터 시작"

### Phase 2: 라벨링 및 저장 (자동화 방식)

** 목표**: 터미널 출력을 최소화하여 토큰 소비 절감 (~6K/배치)

#### 배치 루프 (${BATCH_SIZE}건 단위)
각 배치마다 다음 절차를 반복:

1. **데이터 추출 (출력 안 봄)**:
   ```bash
   # 현재 배치 범위 계산 (예: 51~100)
   BATCH_START = [현재 배치 시작 라인]
   BATCH_END = [현재 배치 종료 라인]
   
   # 파일로 직접 저장 (터미널 출력 X)
   sed -n '${BATCH_START},${BATCH_END}p' ${SOURCE_DATA} > temp_chunk.tsv
   ```

2. **범용 스크립트 준비**:
   - `scripts/process_chunk.py`를 복사하여 `process_batch_N.py` 생성
   - 스크립트 내부의 **에이전트 작업 영역**만 수정:
     ```python
     # 에이전트 작업 영역 (여기만 수정)
     for row in raw_data:
         source = row[0]
         # ${LABELING_RULES}에 따라 라벨링
         target = "..."  # 직접 번역 (AI 감상 금지!)
         action = 0
         severity = 0
         category = "NORMAL"
         reason = "..."
         origin_tags = row[-1]
         
         labeled_data.append([source, target, action, severity, category, reason, origin_tags])
     # 에이전트 작업 영역 끝
     ```

3. **실행 및 자동 검증**:
   ```bash
   python process_batch_N.py temp_chunk.tsv ${BATCH_START} ${BATCH_END}
   ```
   - 스크립트가 자동으로:
     - ✓ 데이터 개수 검증 (예상 vs 실제)
     - ✓ 라벨링 수행
     - ✓ TSV 파일에 Append
     - ✓ 최종 개수 재검증

4. **보고**: "Batch N 완료: Line ${BATCH_START}~${BATCH_END} (${BATCH_SIZE}건)"

#### 절대 준수 사항
- **(금기)** `target`에 AI 평론/훈계/도덕적 거리두기 절대 금지
- **(사족 금지)** "참 씁쓸한 현실입니다" 등 AI 감상 절대 금지
- **(직접 번역)** 원문만큼만 말하기. 부연 설명 금지

### Phase 3: 마무리 및 검증
1. **정리**: 
   ```bash
   rm temp_chunk.tsv process_batch_*.py
   ```
2. **검증**: `wc -l ${OUTPUT_DATA}`로 최종 라인 수 확인
3. **최종 보고**:
   - 범위: Line ${START_LINE} ~ ${END_LINE}
   - 건수: [계산] 건
   - 상태: [전체 라인 수] 줄 완료. 이상 없음 확인.
