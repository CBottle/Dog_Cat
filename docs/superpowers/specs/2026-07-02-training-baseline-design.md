# 02 Training Baseline Design

## Goal

`02_training_baseline.ipynb`는 `01_data_split.ipynb`에서 저장한 split CSV를 사용해 PyTorch 기본 학습 파이프라인을 단계별로 만드는 노트북입니다.

이번 단계의 핵심은 높은 성능을 바로 내는 것이 아니라, Jun이 데이터 로드부터 학습 루프까지 본인의 말로 설명할 수 있게 만드는 것입니다.

## Confirmed Direction

- 노트북은 `01_data_split.ipynb`처럼 번호가 있는 Step 구조로 작성합니다.
- 문장은 `~합니다`, `~했습니다` 체를 사용합니다.
- 어려운 코드보다 설명 가능한 코드를 우선합니다.
- PyTorch 학습 흐름은 직접 작성합니다.
- 기본 실험은 증강 없는 baseline으로 시작합니다.
- OpenCV 증강은 추후 발전 실험으로 분리하고, 기본 데이터셋과 증강 데이터셋을 비교할 수 있게 남겨둡니다.

## Notebook Structure

1. Step 0. 이 노트북을 작성하는 이유
2. Step 1. 이번 학습 실험의 목표 정리
3. Step 2. 필요한 라이브러리 불러오기
4. Step 3. 01에서 저장한 split CSV 불러오기
5. Step 4. 사용할 split 비율 선택하기
6. Step 5. 이미지 전처리 방식 결정하기
7. Step 6. PyTorch Dataset 만들기
8. Step 7. DataLoader 만들기
9. Step 8. baseline 모델 선택하기
10. Step 9. loss function과 optimizer 선택하기
11. Step 10. CPU/GPU device 선택하기
12. Step 11. train 1 epoch 함수 만들기
13. Step 12. validation 함수 만들기
14. Step 13. 전체 학습 루프 만들기
15. Step 14. train/validation loss curve 그리기
16. Step 15. CPU/GPU 학습 시간 비교를 위한 기록 남기기
17. Step 16. 결과를 보고 다음 실험 계획 정리하기

## Model Approach

첫 baseline은 간단한 CNN을 사용합니다.

비교 후보는 다음과 같습니다.

- 간단한 CNN: 이해하기 쉽고 학습 루프를 직접 설명하기 좋습니다.
- pretrained ResNet: 성능은 좋을 수 있지만 처음 학습 흐름을 설명하기에는 복잡합니다.
- 직접 만든 더 깊은 CNN: 구조 설명이 길어지고 초보 단계에서는 과할 수 있습니다.

이번 단계에서는 간단한 CNN을 선택합니다.

## Presentation Learning Note

발표 자료에는 모델을 만들면서 함께 학습한 용어도 접근 과정으로 포함합니다.

예시는 다음과 같습니다.

- DNN은 여러 층을 쌓은 신경망을 넓게 부르는 말입니다.
- CNN은 이미지의 공간적 특징을 학습하기 좋은 DNN 계열 구조입니다.
- ResNet은 CNN을 깊게 쌓으면서 skip connection을 사용한 모델입니다.
- TensorFlow와 PyTorch는 모두 딥러닝 프레임워크이지만, 이번 과제에서는 학습 루프가 코드에 잘 드러나는 PyTorch를 선택했습니다.
- ReLU는 모델 layer 사이에서 사용하는 activation function입니다.
- CrossEntropyLoss는 분류 문제에서 예측과 정답의 차이를 계산하는 loss function입니다.
- Adam은 loss를 줄이는 방향으로 모델 가중치를 업데이트하는 optimizer입니다.

이 내용은 단순한 용어 정리가 아니라, Jun이 왜 CNN과 PyTorch를 선택했는지 설명하는 근거로 사용합니다.

Keras와 비교하면 `Dense(..., activation="relu")`의 `relu`는 PyTorch의 `nn.ReLU()`와 비슷하고, `model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")`의 loss와 optimizer는 PyTorch에서 `criterion = nn.CrossEntropyLoss()`와 `optimizer = optim.Adam(...)`으로 분리해서 정의합니다.

## OpenCV Development Note

OpenCV 증강은 기본 학습을 이해한 뒤 다음 발전 실험으로 진행합니다.

비교 목표는 다음과 같습니다.

- 기본 train 데이터로 학습한 결과
- OpenCV로 조명, 대비, 회전, 흐림 등을 적용한 train 증강 데이터로 학습한 결과

중요한 원칙은 split 이후 train set에만 증강을 적용하는 것입니다. validation과 test를 증강하면 평가 기준이 달라져 공정한 비교가 어려워집니다.

## Success Criteria

- Jun이 각 Step의 목적을 설명할 수 있습니다.
- `optimizer.zero_grad -> forward -> loss -> backward -> step` 흐름이 코드에 보입니다.
- 기본 train/validation loss를 기록할 수 있습니다.
- CPU/GPU 시간 비교를 넣을 자리를 확보합니다.
- OpenCV 증강 비교는 후속 실험으로 자연스럽게 연결됩니다.
