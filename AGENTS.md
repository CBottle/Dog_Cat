# Dog/Cat Classification Agents

이 프로젝트는 Oxford-IIIT Pet Dataset을 사용해 강아지/고양이 이진 이미지 분류 파이프라인을 직접 구축하는 과제입니다.

## Confirmed Decisions

- 방향: 초보자가 설명할 수 있는 균형형 프로젝트
- 데이터셋: Oxford-IIIT Pet Dataset
- 핵심 목표: 높은 점수보다 데이터 분리, 학습 흐름, overfitting 판단, 지표 해석, 오분류 분석을 직접 설명하는 것
- 산출물: 결과 정리 문서와 Day 2 발표 자료

## Agent Map

- Project Mentor: 전체 진행, 의사결정, 초보자 설명
- Data Agent: 데이터 로드, dog/cat 라벨 변환, train/validation/test 분리
- Training Agent: PyTorch 모델과 학습 루프 구현
- Evaluation Agent: accuracy, confusion matrix, precision, recall 계산과 해석
- Error Analysis Agent: 오분류 이미지 확인, 원인 유형화, 개선안 도출
- Report Agent: 결과 정리 문서 작성
- Presentation Agent: 발표 슬라이드 구조와 발표 스크립트 작성

## Role Files

- `.agents/project-mentor.md`
- `.agents/data-agent.md`
- `.agents/training-agent.md`
- `.agents/evaluation-agent.md`
- `.agents/error-analysis-agent.md`
- `.agents/report-agent.md`
- `.agents/presentation-agent.md`

## Operating Rules

1. Jun이 이해하고 발표할 수 있는 설명을 우선한다.
2. 중요한 선택지는 2-3개 보기로 제안하고 추천 이유를 짧게 설명한다.
3. 코드는 직접 쓴 학습 흐름이 보이게 한다.
4. 실험 결과는 항상 발표 문장으로 바꿀 수 있게 정리한다.
5. 데이터 원본, 학습 결과, 모델 체크포인트는 Git에 커밋하지 않는다.
6. 문서, 코드, 설정, 작은 예시는 Git에 커밋한다.

## Project Folders

- `docs/planning/`: 과제 진행 노트와 의사결정 기록
- `docs/reports/`: 최종 결과 정리 문서
- `slides/`: 발표 자료
- `src/`: Python 코드
- `notebooks/`: 실험 노트북
- `data/raw/`: 원본 데이터
- `data/processed/`: 분리/가공된 데이터 또는 split 파일
- `outputs/figures/`: loss curve, confusion matrix 등 그림
- `outputs/metrics/`: metric 결과 파일
- `outputs/misclassified/`: 오분류 이미지 샘플

## Next Workflow

1. 데이터셋 로드 방식 결정
2. dog/cat 라벨 변환 규칙 작성
3. train/validation/test split 생성
4. baseline 학습 코드 작성
5. loss curve와 metric 계산
6. 오분류 이미지 분석
7. 결과 문서와 발표 자료 작성
