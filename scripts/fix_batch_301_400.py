
import csv
import os

path = 'data/processed/clean_lingual_v0.5.tsv'
temp_path = path + '.tmp'

# 사용자 피드백 반영: 1:1 매칭 강화 및 불필요한 의역 제거
fixes = {
    '그래서 이슬람이 싫다 남은 생각하지않고 오직 이슬람만 안다': '그래서 이슬람교가 싫습니다. 남은 생각하지 않고 오직 이슬람교만 압니다.',
    '보지파티는 착한 퀴어어': '여성들만의 파티는 착한 퀴어 행사입니다.'
}

rows = []
modified_count = 0
with open(path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    rows.append(header)
    for row in reader:
        source = row[0]
        if source in fixes:
            row[1] = fixes[source]
            modified_count += 1
        rows.append(row)

with open(temp_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(rows)

os.replace(temp_path, path)
print(f'Batch Fix (301-400): Successfully updated {modified_count} items based on user feedback.')
