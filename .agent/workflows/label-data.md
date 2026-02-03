---
description: 대량 데이터 라벨링 자동화 (Turbo Mode)
---

// turbo-all

# 대량 라벨링 자동화 (SOP v0.7.0)
사용법: `/label-data {START_LINE} {END_LINE}`

## 중요: 원칙 통합 (Source of Truth)
본 워크플로우는 **운영 절차(Operation)**만 담당합니다. 모든 **라벨링 논리(Logic)** 및 **금기 사항**은 아래 두 파일을 100% 따릅니다.
- **실행 로직**: `prompts/labeling_workflow.md` (마스터 SOP)
- **라벨링 규칙**: `prompts/labeling_rules.md` (헌법)

## 1. 전역 설정 (Config)
- **Source**: `data/raw/unsmile_train.tsv`
- **Output**: `data/processed/clean_lingual_v0.5.tsv`
- **Batch**: 50건 단위

## 2. 실행 절차 (Autonomous)

### Phase A: 지침 로드 및 환경 확인
1. **지침 정독**: 반드시 `prompts/labeling_workflow.md`와 `prompts/labeling_rules.md`를 먼저 읽고 최근 업데이트된 **'도덕적 거리두기 금지'** 등의 조항을 숙지하십시오.
2. **상태 확인**: `wc -l {Output}`로 현재 진행 상태 보고.
3. **정밀 추출**: `sed -n '{START_LINE},{END_LINE}p' {Source} > temp_batch.tsv` 실행. (토큰 보호를 위해 `view_file` 금지)

### Phase B: 배치 라벨링 및 적재 (Loop)
1. **작업 수행**: `prompts/labeling_workflow.md`의 **Phase 2** 절차를 무한 루프로 수행합니다.
2. **저장 스크립트 작성**: 반드시 파이썬 `csv` 모듈을 사용하십시오 (pandas 금지).
3. **무중단 실행**: `write_to_file` 후 `python script.py` 실행 (SafeToAutoRun 활성화).

### Phase C: 마무리
1. `rm temp_batch.tsv` 및 작업 스크립트 삭제.
2. 최종 라인 수 보고 및 `prompts/labeling_workflow.md`의 **Phase 3**에 따라 최종 검증.

## 절대 준수 사항
- 모든 작업물은 **`prompts/labeling_rules.md`**의 최신 버전을 기준으로 생성되어야 합니다.
- **사족 절대 금지**: "참 씁쓸한 현실입니다"와 같은 AI 감상평이 섞일 경우 본 워크플로우는 실패한 것으로 간주합니다.
- **무결성**: `origin_tags`는 원본의 값을 100% 복제합니다.