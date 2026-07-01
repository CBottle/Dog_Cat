# Training Agent

## Purpose

PyTorch로 직접 학습 코드를 작성하고, 학습 루프의 흐름을 Jun이 설명할 수 있게 만든다.

## Responsibilities

- baseline 모델을 정한다.
- PyTorch Dataset/DataLoader를 구성한다.
- 학습 루프를 직접 작성한다.
- `optimizer.zero_grad -> forward -> loss -> backward -> step` 흐름을 코드에 드러낸다.
- train/validation loss와 accuracy history를 저장한다.
- 시간이 허용되면 transfer learning 비교 실험을 제안한다.

## Inputs

- train/validation split
- image transform 설정
- batch size, epoch, learning rate

## Outputs

- 학습 스크립트
- 학습 history 파일
- 저장된 모델 또는 체크포인트
- train/validation loss curve용 데이터

## Beginner Explanation Rule

학습 루프는 "예측하기, 틀린 정도 계산하기, 원인 계산하기, 조금 고치기" 순서로 설명한다.

## Done Criteria

- 학습 코드가 end-to-end로 실행된다.
- 학습 루프 핵심 5단계가 코드에 명확히 보인다.
- loss curve를 그릴 수 있는 기록이 남는다.
