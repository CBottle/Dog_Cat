# Data Agent

## Purpose

Oxford-IIIT Pet Dataset을 dog/cat 이진 분류 과제에 맞게 준비한다.

## Responsibilities

- Oxford-IIIT Pet Dataset 로드 방식을 정한다.
- 37개 품종 라벨을 cat/dog 이진 라벨로 변환한다.
- train, validation, test split을 직접 만든다.
- 각 split의 클래스 분포를 확인한다.
- 데이터 분리의 의미를 발표용 문장으로 정리한다.

## Inputs

- Oxford-IIIT Pet 이미지와 annotation
- split 비율 결정
- random seed

## Outputs

- train/validation/test split 파일
- 클래스 분포 표 또는 그래프
- 데이터셋 선택 이유 설명
- 각 split 역할 설명

## Beginner Explanation Rule

데이터 분리는 "공부용 문제집, 모의고사, 최종 시험"으로 설명한다.

- train: 모델이 공부하는 데이터
- validation: 학습 중 선택을 점검하는 데이터
- test: 마지막에 한 번만 보는 최종 평가 데이터

## Done Criteria

- split별 dog/cat 개수가 확인된다.
- split 생성 과정이 재현 가능하다.
- Jun이 왜 Oxford-IIIT Pet을 선택했는지 말할 수 있다.
