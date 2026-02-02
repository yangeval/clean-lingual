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

### Phase 2: 라벨링 및 저장
1. **추출**: `sed -n '${START_LINE},${END_LINE}p' ${SOURCE_DATA}` 사용. 실패 시에만 파이썬 `csv` 모듈로 추출.
2. **수행**: 
   - `${BATCH_SIZE}` 단위로 `${LABELING_RULES}`에 맞춰 라벨링.
   - **(절대 금기)** `target`에 AI 평론/훈계/도덕적 훈수 절대 금지. 
   - **(사족 금지)** "참 씁쓸한 현실입니다" 등 AI의 감상이나 부연 설명을 절대 덧붙이지 마십시오. 원문만큼만 말하십시오. 직접 번역 원칙 준수.
3. **저장**: `write_to_file`로 다음 템플릿의 `.py`를 생성하여 Append (**pandas 금지**).
   ```python
   import csv
   data = [ [source, target, action, severity, category, reason, origin_tags], ... ]
   with open('${OUTPUT_DATA}', 'a', encoding='utf-8', newline='') as f:
       writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
       writer.writerows(data)
   ```
4. **보고**: ${BATCH_SIZE}건마다 "Line [X]~[Y] 완료 (N/Total)"

### Phase 3: 마무리 및 검증
1. **정리**: `rm temp_batch_*.tsv` 로 임시 파일 삭제
2. **검증**: `wc -l ${OUTPUT_DATA}`로 최종 라인 수 확인
3. **최종 보고**:
   - 범위: Line ${START_LINE} ~ ${END_LINE}
   - 건수: [계산] 건
   - 상태: [전체 라인 수] 줄 완료. 이상 없음 확인.
