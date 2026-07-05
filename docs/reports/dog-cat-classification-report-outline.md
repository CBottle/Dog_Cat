# Dog/Cat 이미지 분류 파이프라인 구축 및 분석 보고서 초안

## 0. 문서 목적

이 문서는 Oxford-IIIT Pet Dataset을 사용해 강아지/고양이 이진 이미지 분류 파이프라인을 직접 구축한 과정과 결과를 정리하기 위한 초안입니다.

이번 과제의 핵심은 최고 성능 모델을 만드는 것보다, 데이터 분리, PyTorch 학습 루프, overfitting 판단, precision/recall 해석, 오분류 분석 과정을 본인 언어로 설명하는 것입니다.

## 1. 요구사항 대응표

| 요구사항 | 문서 위치 | 포함할 핵심 내용 |
|---|---|---|
| train / validation / test 직접 구성 및 역할 설명 | 2장 | Oxford-IIIT Pet Dataset 선택 이유, 70/15/15 split, 각 데이터셋 역할 |
| PyTorch 직접 학습 코드 작성 | 3장 | `optimizer.zero_grad -> forward -> loss -> backward -> step` 흐름 |
| train/val loss curve와 overfitting 판단 | 5장 | baseline, OpenCV, regularized 모델의 loss curve 비교 |
| accuracy, confusion matrix, precision/recall 계산과 해석 | 6장 | test set 기준 지표 비교와 상황별 중요 지표 정리 |
| 오분류 이미지 확인, 원인 규명, 방안 도출 | 7장, 8장 | confidence 높은 오분류 확인, 실패 유형 분류, 개선 방향 |

## 2. 프로젝트 개요

### 2.1 과제 목적

- 강아지/고양이 이미지 분류 파이프라인을 직접 구축합니다.
- 면접 공통 질문인 데이터 분포, overfitting, precision/recall을 실습으로 검증합니다.
- 실무에서 중요한 실패 데이터 유형 분류와 패턴 도출 과정을 미니 버전으로 경험합니다.

### 2.2 사용 데이터셋

- 데이터셋: Oxford-IIIT Pet Dataset
- 구성: 37개 품종, 강아지/고양이 이미지
- 사용 방식: 품종 분류가 아니라 dog/cat 이진 분류로 사용합니다.

### 2.3 Oxford-IIIT Pet Dataset을 선택한 이유

- Kaggle Cat and Dog 데이터셋은 train/test가 이미 나누어져 있어 편리하지만, 이번 과제에서는 직접 train/validation/test를 나누는 경험이 중요합니다.
- Oxford-IIIT Pet Dataset은 dog/cat 라벨뿐 아니라 품종 라벨도 가지고 있어, 나중에 품종 분류로 확장할 수 있습니다.
- breed 기준으로 데이터를 나누면 이진 분류를 하더라도 다양한 품종이 train/validation/test에 고르게 들어가도록 설계할 수 있습니다.

## 3. 데이터 분리 전략

### 3.1 라벨 구성

이번 프로젝트에서는 `torchvision.datasets.OxfordIIITPet`에서 제공하는 binary label 정보를 사용했습니다.

정리할 내용:

- 이미지 경로
- dog/cat 라벨
- 품종 이름
- split 정보

### 3.2 train / validation / test 비율

최종 실험에서는 `70 / 15 / 15` 비율을 사용했습니다.

| split | 역할 |
|---|---|
| train | 모델이 실제로 학습하는 데이터입니다. 가중치가 업데이트됩니다. |
| validation | 학습 중 모델 선택, overfitting 판단, early stopping에 참고하는 데이터입니다. 가중치는 업데이트되지 않습니다. |
| test | 최종 모델을 평가하기 위한 데이터입니다. 학습과 모델 선택에 사용하지 않습니다. |

### 3.3 breed 기준 stratified split을 사용한 이유

단순히 랜덤으로 나누면 특정 품종이 train에만 많이 들어가고 test에는 적게 들어갈 수 있습니다.

이번 프로젝트는 dog/cat 이진 분류이지만, 품종별 특성이 dog/cat 판단에 영향을 줄 수 있으므로 breed 기준으로 고르게 나누는 것이 더 적절하다고 판단했습니다.

발표용 문장:

> 이진 분류만 한다면 dog/cat 비율만 맞추면 된다고 생각할 수 있지만, 실제 이미지에서는 품종에 따라 생김새가 크게 다릅니다. 그래서 특정 품종이 한 split에 치우치지 않도록 breed 기준 stratified split을 사용했습니다.

## 4. PyTorch 학습 파이프라인

### 4.1 PyTorch를 사용한 이유

TensorFlow/Keras의 `model.fit()`은 학습이 간단하지만 내부 학습 흐름이 많이 감춰집니다.

이번 과제에서는 학습 과정을 직접 이해하는 것이 중요하므로 PyTorch로 학습 루프를 직접 작성했습니다.

### 4.2 직접 작성한 학습 루프

학습 흐름은 다음과 같습니다.

```python
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

각 단계의 의미:

| 코드 | 의미 |
|---|---|
| `optimizer.zero_grad()` | 이전 batch에서 계산된 gradient를 지웁니다. |
| `outputs = model(images)` | 모델이 이미지를 보고 예측합니다. forward 단계입니다. |
| `loss = criterion(outputs, labels)` | 예측과 정답을 비교해 오차를 계산합니다. |
| `loss.backward()` | 역전파를 통해 각 가중치가 얼마나 수정되어야 하는지 gradient를 계산합니다. |
| `optimizer.step()` | 계산된 gradient를 사용해 실제 가중치를 업데이트합니다. |

발표용 문장:

> PyTorch는 Keras의 fit처럼 한 줄로 학습을 끝내지 않고, zero_grad, forward, loss, backward, step 흐름을 직접 작성해야 합니다. 덕분에 모델이 예측하고, 오차를 계산하고, 오차를 줄이는 방향으로 가중치를 업데이트하는 과정을 직접 확인할 수 있었습니다.

## 5. 모델 실험 설계

이번 프로젝트에서는 한 번에 완성 모델을 만들기보다, 하나씩 개선하면서 비교했습니다.

| 실험 | 데이터 | 모델/개선점 | 목적 |
|---|---|---|---|
| 실험 1 | 원본 train | Baseline 3 Epoch SimpleCNN | 가장 기본적인 기준 성능 확인 |
| 실험 2 | 원본 train | Baseline + EarlyStopping | epoch를 늘렸을 때 개선되는지 확인 |
| 실험 3 | OpenCV 증강 train | SimpleCNN | 데이터 증강만으로 개선되는지 확인 |
| 실험 4 | 원본 train | Regularized CNN | 규제가 overfitting 완화에 도움이 되는지 확인 |
| 실험 5 | OpenCV 증강 train | Regularized CNN | 증강과 규제를 함께 적용했을 때 효과 확인 |

### 5.1 Baseline SimpleCNN

처음에는 단순 CNN을 사용해 기준 모델을 만들었습니다.

주요 목적:

- PyTorch 학습 루프 확인
- train/validation loss curve 확인
- overfitting이 발생하는지 관찰

### 5.2 OpenCV 증강 실험

OpenCV로 train 이미지만 증강했습니다.

사용한 증강:

- 밝기 조정
- 대비 조정
- blur
- 회전
- 좌우 반전

결과 해석:

> OpenCV 증강으로 train 데이터 수를 늘렸지만, 증강만으로는 overfitting이 충분히 해결되지 않았습니다. 이는 비슷한 원본 이미지의 변형이 많아졌을 뿐, validation/test 분포에 실제로 도움이 되는 다양성이 충분하지 않았을 가능성이 있습니다.

### 5.3 Regularized CNN

overfitting을 줄이기 위해 다음 방법을 추가했습니다.

- BatchNorm: 학습 안정화
- Dropout: 일부 뉴런을 임시로 꺼서 train 데이터 암기 완화
- weight decay: 가중치가 너무 커지는 것 억제
- 낮춘 learning rate: 너무 빠른 과적합 완화
- LR Scheduler: validation loss가 정체되면 learning rate 감소

## 6. 모델 비교와 최종 선택

### 6.1 전체 모델 비교 기준

모델 선택 시 하나의 숫자만 보지 않고 다음 기준을 함께 봤습니다.

- best validation loss
- best validation accuracy
- final validation accuracy
- 학습 시간
- 모델 구조의 설명 가능성
- 데이터셋 특성

### 6.2 최종 선택 모델

최종 선택 후보는 `Regularized Original` 모델입니다.

선택 이유:

- baseline보다 validation loss와 accuracy가 개선되었습니다.
- OpenCV 증강 모델보다 학습 시간이 훨씬 짧습니다.
- OpenCV 증강 모델은 best validation loss가 조금 더 낮았지만 accuracy 차이가 크지 않았고, 학습 시간이 크게 늘었습니다.
- Oxford-IIIT Pet Dataset은 비교적 좋은 조건에서 촬영된 이미지가 많아, 조명 변화나 blur 중심의 증강 효과가 제한적이었을 수 있습니다.

발표용 문장:

> OpenCV 증강 모델의 best validation loss가 조금 더 낮았지만, accuracy 차이는 크지 않았고 학습 시간은 훨씬 길었습니다. 따라서 성능과 효율의 균형을 고려해 Regularized Original 모델을 최종 선택 후보로 정했습니다.

## 7. Loss Curve와 Overfitting 판단

### 7.1 train loss와 validation loss의 의미

- train loss: 모델이 학습한 데이터에서의 오차입니다.
- validation loss: 학습에 직접 사용하지 않은 데이터에서의 오차입니다.

### 7.2 overfitting 판단 기준

overfitting은 train loss는 계속 낮아지지만 validation loss가 다시 올라가는 패턴으로 판단했습니다.

해석:

> train 데이터에는 점점 잘 맞지만, 새로운 데이터인 validation 데이터에는 오히려 약해졌다는 의미입니다. 이는 모델이 일반적인 dog/cat 특징을 배운 것이 아니라 train 데이터의 특정 패턴까지 외우기 시작했을 가능성을 보여줍니다.

### 7.3 실험별 관찰

- Baseline: train loss는 낮아졌지만 validation loss가 올라가 overfitting 신호가 나타났습니다.
- OpenCV-only: 데이터 수는 늘었지만 overfitting이 충분히 해결되지 않았습니다.
- Regularized CNN: validation loss가 더 안정적으로 낮아져 overfitting 완화 효과가 있었습니다.

## 8. 최종 평가 지표 분석

### 8.1 accuracy

전체 test 이미지 중 모델이 맞힌 비율입니다.

장점:

- 직관적입니다.
- 데이터가 균형적일 때 기본 비교 지표로 좋습니다.

한계:

- 어떤 class에서 틀렸는지 알기 어렵습니다.

### 8.2 confusion matrix

모델이 어떤 방향으로 틀렸는지 보여줍니다.

예를 들어:

- 실제 cat을 dog로 예측한 경우
- 실제 dog를 cat으로 예측한 경우

### 8.3 precision

모델이 특정 class라고 예측한 것 중 실제로 맞은 비율입니다.

예시:

```text
cat precision = 맞게 cat이라고 예측한 수 / cat이라고 예측한 전체 수
```

precision이 중요한 상황:

- 모델이 예측한 결과를 얼마나 믿을 수 있는지가 중요할 때
- 잘못된 자동 태그를 줄이고 싶을 때

### 8.4 recall

실제 특정 class 중 모델이 찾아낸 비율입니다.

예시:

```text
cat recall = 맞게 cat이라고 예측한 수 / 실제 cat 전체 수
```

recall이 중요한 상황:

- 특정 class를 놓치지 않는 것이 중요할 때
- 실제 dog 이미지를 최대한 빠짐없이 찾아야 할 때

### 8.5 macro average

cat과 dog를 같은 비중으로 보고 평균낸 값입니다.

발표용 문장:

> accuracy는 전체 정답률을 보여주지만, 어떤 class에서 어떤 방향으로 틀렸는지는 알 수 없습니다. 그래서 confusion matrix로 오분류 방향을 확인하고, precision과 recall을 함께 계산했습니다. precision은 예측의 신뢰도가 중요할 때, recall은 놓치지 않는 것이 중요할 때 더 의미가 있다고 판단했습니다.

## 9. 오분류 이미지 분석

### 9.1 분석 방법

1. test set에서 예측을 수행했습니다.
2. 실제 라벨과 예측 라벨이 다른 이미지를 추출했습니다.
3. confidence가 높은 오분류부터 확인했습니다.
4. 이미지, 실제 라벨, 예측 라벨, confidence, breed를 함께 확인했습니다.
5. 반복되는 실패 유형을 분류했습니다.

### 9.2 confidence를 함께 본 이유

confidence는 모델이 자기 예측을 얼마나 확신했는지를 의미합니다.

- confidence가 낮은 오분류: 모델도 애매하게 헷갈린 경우
- confidence가 높은 오분류: 모델이 잘못된 특징을 강하게 믿은 경우

### 9.3 주요 오분류 원인 후보

| 원인 유형 | 설명 | 개선 방향 |
|---|---|---|
| 얼굴 일부만 보임 | 눈, 귀, 코 등 핵심 특징이 부족함 | head ROI 활용, 더 큰 입력 이미지 |
| 배경이 복잡함 | 모델이 배경을 잘못된 근거로 사용할 수 있음 | segmentation 활용 |
| 품종 특성이 애매함 | 특정 강아지/고양이 품종이 서로 비슷하게 보임 | 품종별 오분류 분석 |
| 어둡거나 흐림 | 특징이 잘 보이지 않음 | 실패 유형에 맞춘 밝기/blur 증강 |
| 동물이 작게 찍힘 | 객체 특징보다 배경 비중이 큼 | crop, ROI, segmentation |

발표용 문장:

> 오분류 이미지를 직접 확인한 결과, 모델이 무작위로 틀린 것이 아니라 얼굴 일부만 보이는 경우, 배경이 복잡한 경우, 품종 특성이 애매한 경우처럼 반복되는 실패 유형이 있었습니다. 이 과정을 통해 단순 점수 개선보다 실패 유형을 분류하고 개선 방향을 찾는 것이 실무적으로 중요하다고 느꼈습니다.

## 10. 향후 개선 방향

### 10.1 head ROI 활용

Oxford-IIIT Pet Dataset에는 동물의 머리 위치 정보가 포함되어 있습니다.

dog/cat 구분에는 얼굴, 귀, 눈, 코 주변 특징이 중요하므로, head ROI를 활용하면 모델이 더 중요한 영역을 볼 수 있습니다.

### 10.2 segmentation 활용

배경이 복잡한 이미지에서 모델이 배경을 잘못된 근거로 사용할 수 있습니다.

segmentation annotation을 활용하면 동물 영역에 더 집중하도록 만들 수 있습니다.

### 10.3 실패 유형에 맞춘 데이터 증강

OpenCV 증강을 무작정 많이 적용하는 것보다, 오분류 원인에 맞춰 증강하는 것이 더 적절합니다.

예시:

- 어두운 사진에서 많이 틀리면 밝기/대비 증강
- 흐린 사진에서 많이 틀리면 blur 대응
- 위치 변화에서 많이 틀리면 crop/rotate 조정

### 10.4 전이학습 적용

ResNet, MobileNet, EfficientNet처럼 ImageNet에서 사전 학습된 모델을 활용할 수 있습니다.

직접 만든 CNN은 학습 과정을 설명하기 좋지만, 더 높은 성능을 목표로 한다면 사전 학습 모델을 사용하는 전이학습이 효과적일 수 있습니다.

### 10.5 CNN 추가 튜닝

직접 만든 CNN 안에서도 다음 실험을 해볼 수 있습니다.

- Flatten 대신 AdaptiveAvgPool 사용
- Adam 대신 AdamW 비교
- learning rate 조정
- batch size 비교
- 입력 이미지 크기 확대

## 11. 결론

이번 프로젝트에서는 단순히 모델을 학습하는 것에서 끝내지 않고, 데이터 분리, 학습 흐름, overfitting 판단, 지표 해석, 오분류 분석까지 전체 과정을 직접 수행했습니다.

최종 모델은 성능과 학습 시간의 균형을 고려해 Regularized Original 모델을 선택했습니다.

이번 실습을 통해 accuracy 하나만으로 모델을 판단하기 어렵고, confusion matrix, precision/recall, 오분류 이미지 분석까지 함께 봐야 모델의 실제 실패 원인을 이해할 수 있다는 점을 확인했습니다.

최종 발표용 결론:

> 이번 과제에서는 PyTorch로 dog/cat 이미지 분류 파이프라인을 직접 구축했습니다. train/validation/test를 직접 나누고, baseline부터 OpenCV 증강, 규제 모델까지 단계적으로 실험했습니다. loss curve를 통해 overfitting을 판단했고, 최종 모델은 accuracy뿐 아니라 confusion matrix, precision/recall, 오분류 이미지까지 분석했습니다. 결과적으로 완벽한 모델을 만드는 것보다 모델이 왜 틀리는지 확인하고 개선 방향을 도출하는 과정이 더 중요하다는 것을 확인했습니다.
