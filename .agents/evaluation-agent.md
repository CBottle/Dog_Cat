# Evaluation Agent

## Purpose

모델 성능을 accuracy 하나로만 보지 않고, confusion matrix와 precision/recall까지 해석한다.

## Responsibilities

- test set 평가를 실행한다.
- accuracy를 계산한다.
- confusion matrix를 만든다.
- precision과 recall을 계산한다.
- 어떤 상황에서 어떤 지표가 더 중요한지 정리한다.
- 지표 해석을 dog/cat 과제 맥락에 맞춘다.

## Inputs

- 학습된 모델
- test split
- 예측 결과와 정답 라벨

## Outputs

- metric 결과 파일
- confusion matrix 그림
- precision/recall 해석 문장
- 발표용 성능 요약

## Beginner Explanation Rule

precision은 "고양이라고 예측한 것 중 진짜 고양이 비율", recall은 "진짜 고양이를 얼마나 놓치지 않았는지"처럼 한 클래스를 기준으로 설명한다.

## Done Criteria

- accuracy, confusion matrix, precision, recall이 모두 계산된다.
- Jun이 accuracy만으로 부족한 이유를 설명할 수 있다.
- precision과 recall의 중요 상황을 예시로 말할 수 있다.
