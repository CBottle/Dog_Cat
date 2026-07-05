# Dog/Cat 분류 프로젝트 의사결정 기록

## 목적

이 문서는 강아지/고양이 이미지 분류 과제를 진행하면서 Jun과 함께 고민하고 결정한 내용을 정리한 기록입니다.

최종 목적은 단순히 높은 정확도를 내는 것이 아니라, **어떤 문제를 보고, 어떤 가설을 세우고, 어떤 실험을 추가했으며, 결과를 어떻게 비교했는지** 설명할 수 있게 만드는 것입니다.

## 전체 방향

이번 프로젝트는 완성된 정답 모델을 빠르게 가져오는 것보다, 직접 만든 baseline 모델을 기준으로 하나씩 개선해보는 방향으로 진행합니다.

따라서 pretrained ResNet 같은 전이학습 모델은 당장 사용하지 않고, 직접 만든 CNN 안에서 성능을 개선하는 실험을 우선합니다.

이 방향을 선택한 이유는 다음과 같습니다.

- 과제 요구사항이 PyTorch 학습 흐름을 직접 작성하는 데 초점이 있습니다.
- `optimizer.zero_grad -> forward -> loss -> backward -> step` 흐름을 본인이 설명해야 합니다.
- baseline에서 하나씩 추가해야 overfitting, 데이터 증강, 학습 안정화 효과를 비교하기 쉽습니다.
- 발표에서 "좋은 모델을 가져왔다"보다 "문제를 나누어 실험했다"는 접근 과정이 더 잘 드러납니다.

## 데이터셋 결정

데이터셋은 Oxford-IIIT Pet Dataset을 사용합니다.

선택 이유는 다음과 같습니다.

- 로그인 없이 다운로드할 수 있습니다.
- 강아지/고양이 이진 라벨을 만들 수 있습니다.
- 37개 품종 라벨도 함께 존재합니다.
- 지금은 dog/cat 이진 분류를 하지만, 나중에 품종 분류로 확장할 수 있습니다.
- Kaggle Cat and Dog 데이터셋보다 train/validation/test를 직접 나누는 과정을 보여주기 좋습니다.

## 라벨 생성 결정

Oxford-IIIT Pet Dataset은 품종 이름 첫 글자의 대문자/소문자 규칙으로 dog/cat을 추론할 수도 있습니다.

하지만 이번 프로젝트에서는 `torchvision.datasets.OxfordIIITPet`이 제공하는 binary label 정보를 사용합니다.

사용한 정보는 다음과 같습니다.

- `dataset._images`: 이미지 경로 목록입니다.
- `dataset._labels`: 품종 라벨 번호 목록입니다.
- `dataset._bin_labels`: dog/cat 이진 라벨 번호 목록입니다.
- `dataset.classes`: 품종 번호를 품종 이름으로 바꾸는 목록입니다.
- `dataset.bin_classes`: 이진 라벨 번호를 `Cat`, `Dog` 이름으로 바꾸는 목록입니다.

이 방식을 선택한 이유는 문자열 규칙보다 데이터셋에 포함된 명시적인 라벨을 사용하는 것이 더 안전하기 때문입니다.

## split 결정

train/validation/test split은 직접 구성합니다.

기본 비교 split은 `70/15/15`를 사용합니다.

추가로 `80/10/10`, `60/20/20` split도 만들어두어, 필요하면 split 비율에 따른 결과 차이를 비교할 수 있게 합니다.

split은 단순 dog/cat 기준만 보지 않고, **breed 기준 stratified split**을 사용합니다.

이렇게 결정한 이유는 다음과 같습니다.

- dog/cat 비율만 맞추면 특정 품종이 train에 몰릴 수 있습니다.
- 특정 품종이 test에 너무 적으면 모델이 어떤 품종에 약한지 분석하기 어렵습니다.
- breed 기준으로 나누면 이진 분류에서도 더 다양한 품종을 train/validation/test에 분산시킬 수 있습니다.
- 나중에 특정 품종에서 오분류가 많을 때 데이터 부족 문제인지 확인할 수 있습니다.

## 학습 환경 결정

PyTorch를 사용합니다.

Keras/TensorFlow의 `model.fit()`은 학습 과정을 간단하게 숨겨주지만, 이번 과제에서는 학습 흐름을 직접 보여주는 것이 중요합니다.

따라서 PyTorch로 다음 흐름을 직접 작성합니다.

```text
optimizer.zero_grad
-> forward
-> loss
-> backward
-> optimizer.step
```

GPU가 있으므로 GPU 학습을 사용합니다. 또한 CPU와 GPU 학습 시간 차이도 비교 대상으로 남깁니다.

## 현재까지의 노트북 구조

현재 노트북 흐름은 다음과 같습니다.

```text
01_data_split.ipynb
-> 데이터셋 선택, 라벨 생성, train/validation/test split 생성

02_training_baseline.ipynb
-> 원본 train 데이터로 SimpleCNN baseline 학습

02_1_opencv_augmentation_training.ipynb
-> train 데이터만 OpenCV로 증강해서 SimpleCNN 학습
```

앞으로 추가할 수 있는 노트북은 다음과 같습니다.

```text
02_2_regularized_cnn_training.ipynb
-> 원본 train 데이터에서 규제 모델을 먼저 테스트하고, 이후 OpenCV 증강 train 데이터에 같은 규제 모델을 적용하는 실험

04_evaluation_comparison.ipynb
-> 여러 best model을 같은 test set으로 비교

05_error_analysis.ipynb
-> 오분류 이미지 확인, 원인 유형화, 개선안 정리
```

## 실험 결정 트리

최종 비교 실험은 decision tree처럼 하나씩 추가하는 방식으로 구성합니다.

```text
실험 1: Baseline
실험 2: Baseline + OpenCV
실험 3: Baseline + BatchNorm
실험 4: Baseline + BatchNorm + Dropout
실험 5: Baseline + BatchNorm + Dropout + LR Scheduler
실험 6: 최종 후보 = OpenCV + Improved CNN
```

## 실험별 의도

| 실험 | 추가한 것 | 확인하고 싶은 것 |
|---|---|---|
| 실험 1. Baseline | 없음 | 직접 만든 기본 CNN의 기준 성능을 확인합니다. |
| 실험 2. Baseline + OpenCV | train 데이터 증강 | 데이터 다양성을 늘리면 일반화 성능이 좋아지는지 확인합니다. |
| 실험 3. Baseline + BatchNorm | BatchNorm | 학습이 더 안정적으로 진행되는지 확인합니다. |
| 실험 4. Baseline + BatchNorm + Dropout | Dropout | overfitting이 줄어드는지 확인합니다. |
| 실험 5. Baseline + BatchNorm + Dropout + LR Scheduler | LR Scheduler | validation loss가 정체될 때 learning rate 조절이 도움이 되는지 확인합니다. |
| 실험 6. OpenCV + Improved CNN | 효과가 있었던 개선 조합 | 최종 후보 모델로 가장 좋은 성능을 낼 수 있는지 확인합니다. |

## 왜 하나씩 추가하는지

한 번에 여러 기능을 추가하면 성능이 좋아져도 무엇 때문에 좋아졌는지 설명하기 어렵습니다.

그래서 이번 프로젝트에서는 한 단계마다 하나의 가설을 추가합니다.

예를 들면 다음과 같습니다.

- OpenCV 증강을 추가하면 다양한 촬영 조건에 강해질 것이라고 가정합니다.
- BatchNorm을 추가하면 학습이 더 안정적으로 진행될 것이라고 가정합니다.
- Dropout을 추가하면 train 데이터에만 과하게 맞는 overfitting이 줄어들 것이라고 가정합니다.
- LR Scheduler를 추가하면 validation loss가 정체될 때 더 세밀하게 학습할 수 있을 것이라고 가정합니다.

이 방식은 발표에서 다음 문장으로 설명할 수 있습니다.

> 성능 개선을 한 번에 적용하지 않고, 데이터 개선과 모델 개선 요소를 하나씩 추가했습니다. 이렇게 해야 어떤 선택이 실제로 validation loss, accuracy, overfitting에 영향을 주었는지 비교할 수 있다고 판단했습니다.

## best model 저장 결정

각 실험마다 validation loss 기준 best model을 따로 저장합니다.

예상 checkpoint 파일은 다음과 같습니다.

```text
outputs/checkpoints/
├─ best_01_baseline_simplecnn.pt
├─ best_02_opencv_augmented_simplecnn.pt
├─ best_03_batchnorm_simplecnn.pt
├─ best_04_batchnorm_dropout_simplecnn.pt
├─ best_05_batchnorm_dropout_scheduler_simplecnn.pt
└─ best_06_final_opencv_improved_simplecnn.pt
```

이렇게 저장하는 이유는 나중에 다시 학습하지 않고도 같은 test set에서 여러 모델을 비교할 수 있기 때문입니다.

모델 checkpoint 파일은 실행 결과물이므로 Git에는 커밋하지 않습니다.

## 비교 지표 결정

최종 비교에서는 accuracy만 보지 않습니다.

다음 지표와 결과를 함께 확인합니다.

- train loss / validation loss curve
- accuracy
- confusion matrix
- precision
- recall
- 학습 시간
- 오분류 이미지 샘플
- 오분류 원인 유형

각 지표를 함께 보는 이유는 다음과 같습니다.

- loss curve는 overfitting 여부를 판단하는 데 도움이 됩니다.
- accuracy는 전체적으로 맞춘 비율을 보여줍니다.
- confusion matrix는 cat을 dog로 착각했는지, dog를 cat으로 착각했는지 보여줍니다.
- precision/recall은 어떤 상황에서 어떤 오류가 더 중요한지 설명하는 데 도움이 됩니다.
- 오분류 이미지는 숫자 지표만으로 보이지 않는 실패 패턴을 찾는 데 도움이 됩니다.

## 최종 분석 방향

최종 분석에서는 가장 accuracy가 높은 모델만 고르지 않습니다.

모델별로 다음 질문에 답하는 방식으로 정리합니다.

| 모델 | 분석 질문 |
|---|---|
| Baseline | 기본 CNN만으로 어느 정도 성능이 나왔는가 |
| OpenCV 증강 | 데이터를 늘리면 일반화 성능이 좋아졌는가 |
| BatchNorm | 학습이 더 안정적으로 되었는가 |
| Dropout | overfitting이 줄었는가 |
| LR Scheduler | validation loss 정체 구간에서 추가 개선이 있었는가 |
| Final | 효과 있었던 방법을 합쳤을 때 가장 좋은 결과가 나왔는가 |

## MLOps 관점에서 저장할 것

이번 프로젝트는 완전한 MLOps 시스템은 아니지만, 실험을 재현하고 비교할 수 있게 다음 항목을 저장합니다.

```text
Git에 저장합니다.
- 노트북
- 문서
- requirements.txt
- split 생성 코드
- 실험 설계 기록

Git에 저장하지 않습니다.
- 원본 데이터셋
- 증강 이미지
- 모델 checkpoint .pt 파일
- 큰 출력 결과물
```

실험 결과로는 다음 파일들을 저장합니다.

```text
outputs/checkpoints/
-> best model .pt 파일

outputs/figures/
-> loss curve, confusion matrix 이미지

outputs/metrics/
-> metric CSV, 실험 요약 CSV

outputs/misclassified/
-> 오분류 이미지 샘플
```

## 발표용 핵심 문장

발표에서는 다음 흐름으로 설명합니다.

> 저는 먼저 직접 만든 SimpleCNN을 baseline으로 만들었습니다. 이후 성능 개선을 데이터 개선과 모델 개선으로 나누어 실험했습니다. 데이터 개선으로는 OpenCV 증강을 적용했고, 모델 개선으로는 BatchNorm, Dropout, LR Scheduler를 하나씩 추가했습니다. 각 단계마다 best model을 저장하고, 같은 test set에서 loss curve, accuracy, precision/recall, confusion matrix, 오분류 이미지를 비교하여 어떤 개선이 실제로 도움이 되었는지 판단했습니다.

## 현재 결론

현재까지의 방향은 다음과 같습니다.

- 전이학습은 아직 사용하지 않습니다.
- 직접 만든 CNN baseline을 중심으로 개선합니다.
- 실험은 한 번에 하나씩 추가하는 decision tree 방식으로 진행합니다.
- 각 실험의 best model을 저장합니다.
- 마지막에는 여러 모델을 같은 test set에서 비교합니다.
- 결과 분석은 숫자 지표와 오분류 이미지 분석을 함께 사용합니다.

## 발표 흐름 업데이트

2026-07-05 기준으로 발표 흐름은 다음처럼 가져가는 것이 좋습니다.

```text
1. Baseline 모델을 먼저 만듭니다.
2. epoch를 늘려서 baseline 모델이 어떻게 학습되는지 확인합니다.
3. loss curve를 보고 overfitting 가능성을 판단합니다.
4. 첫 번째 개선책으로 OpenCV 데이터 증강을 적용합니다.
5. OpenCV 증강만으로 충분하지 않으면 규제 방법을 추가합니다.
6. 이후 best model들을 같은 test set에서 비교합니다.
```

발표 키워드는 다음과 같습니다.

```text
Baseline -> Epoch 늘리기 -> OpenCV 증강 -> 규제 -> 최종 비교
```

이 흐름을 선택한 이유는 다음과 같습니다.

- 처음부터 좋은 모델을 만드는 것이 아니라, 문제를 발견하고 개선해가는 과정이 보입니다.
- epoch를 늘리면서 train loss와 validation loss의 차이를 확인할 수 있습니다.
- OpenCV 증강은 데이터 다양성 관점의 개선책입니다.
- 규제는 모델이 train 데이터를 과하게 외우는 문제를 줄이기 위한 개선책입니다.
- 각 단계의 결과가 다음 실험의 이유가 되므로 발표 흐름이 자연스럽습니다.

## 02-1 OpenCV 실험 실행 메모

02-1 OpenCV 증강 실험에서 다음과 같은 결과 흐름을 확인했습니다.

```text
epoch 1/30 | train loss 0.5286, train acc 0.7378 | val loss 0.5196, val acc 0.7663
epoch 2/30 | train loss 0.3259, train acc 0.8629 | val loss 0.6920, val acc 0.7681
epoch 3/30 | train loss 0.1530, train acc 0.9438 | val loss 0.7255, val acc 0.7409
epoch 4/30 | train loss 0.0590, train acc 0.9807 | val loss 0.9627, val acc 0.7717
epoch 5/30 | train loss 0.0303, train acc 0.9899 | val loss 1.2286, val acc 0.7301
epoch 6/30 | train loss 0.0254, train acc 0.9918 | val loss 1.4454, val acc 0.7717
```

이 결과는 OpenCV 증강을 적용했는데도 overfitting이 강하게 나타난 사례로 볼 수 있습니다.

관찰한 점은 다음과 같습니다.

- train loss는 빠르게 낮아졌습니다.
- train accuracy는 99% 수준까지 올라갔습니다.
- validation accuracy는 약 0.73-0.77 근처에서 크게 개선되지 않았습니다.
- validation loss는 epoch가 진행될수록 계속 증가했습니다.
- best model은 validation loss가 가장 낮았던 epoch 1 기준으로 저장되었습니다.

해석은 다음과 같습니다.

- 모델이 train 데이터와 증강 데이터를 빠르게 외웠을 가능성이 있습니다.
- OpenCV 증강이 데이터 수는 늘렸지만, 실제로는 같은 원본 이미지의 변형이므로 모델이 다양성을 충분히 학습하지 못했을 수 있습니다.
- validation loss가 증가한 것은 틀린 예측을 더 강한 확신으로 하는 경우가 늘어난 것으로 볼 수 있습니다.
- OpenCV 증강만으로는 overfitting 완화에 충분하지 않았습니다.

다음 개선 방안은 다음과 같습니다.

- 증강 개수를 줄여서 과도한 유사 이미지 반복을 줄입니다.
- Dropout을 추가해 train 데이터에 과하게 맞는 현상을 줄입니다.
- weight decay를 추가해 가중치가 너무 커지는 것을 억제합니다.
- learning rate를 낮춰 학습이 너무 빠르게 train 데이터에 맞춰지는지 확인합니다.
- BatchNorm을 추가해 학습 안정성을 확인합니다.
- LR Scheduler를 추가해 validation loss가 정체될 때 learning rate를 줄입니다.

발표에서는 다음과 같이 설명할 수 있습니다.

> OpenCV 증강을 적용하면 데이터 다양성이 늘어나 overfitting이 줄어들 것으로 예상했습니다. 하지만 실험 결과 train accuracy는 빠르게 99%까지 상승한 반면 validation loss는 계속 증가했습니다. 이를 통해 단순 증강만으로는 일반화 성능 개선이 충분하지 않다고 판단했고, 다음 실험에서는 Dropout, weight decay, learning rate 조정 같은 규제 방법을 추가로 적용하기로 했습니다.

## 02-2 규제 실험 진행 원칙

02-2는 `02_2_regularized_cnn_training.ipynb`에서 진행합니다.

이 노트북은 다음 원칙으로 구성합니다.

```text
1. 원본 train 데이터로 규제 모델을 먼저 테스트합니다.
2. 원본 데이터에서 규제 효과를 확인한 뒤 OpenCV 증강 train 데이터로 같은 모델을 테스트합니다.
3. OpenCV 증강 train에서도 과적합이 반복되면 증강 이미지 수를 줄입니다.
```

이 순서를 선택한 이유는 다음과 같습니다.

- OpenCV 증강 데이터부터 바로 사용하면 좋아지거나 나빠진 이유를 분리하기 어렵습니다.
- 원본 train에서 규제 모델이 효과가 있는지 먼저 확인해야 합니다.
- OpenCV 증강은 데이터 수를 늘리지만, 같은 원본 이미지의 변형이 많아지면 모델이 그 변형까지 외울 수 있습니다.
- 따라서 OpenCV 데이터에서 과적합이 다시 나타나면 증강 종류를 줄이는 실험이 필요합니다.

02-2에서 추가하는 규제 방법은 다음과 같습니다.

- BatchNorm: 학습 안정화
- Dropout: train 데이터 암기 완화
- weight decay: 가중치가 너무 커지는 것을 억제
- 낮춘 learning rate: 빠른 과적합 완화
- LR Scheduler: validation loss 정체 시 learning rate 감소

발표에서는 다음과 같이 설명할 수 있습니다.

> OpenCV 증강만으로는 overfitting이 충분히 줄어들지 않았습니다. 그래서 다음 단계에서는 모델 쪽 규제 방법을 적용했습니다. 먼저 원본 train 데이터에서 규제 모델의 효과를 확인하고, 이후 같은 규제 모델을 OpenCV 증강 데이터에도 적용했습니다. 만약 OpenCV 데이터에서도 과적합이 반복된다면, 증강 이미지 수를 줄여 유사 이미지가 지나치게 많아지는 문제를 줄이는 방향으로 실험을 이어갑니다.

## 최종 평가와 심층 분석 전략

여러 실험을 모두 같은 깊이로 분석하면 시간이 오래 걸리고 발표 흐름도 복잡해질 수 있습니다.

따라서 먼저 각 실험의 loss와 accuracy를 비교해 좋은 후보 모델을 고릅니다.

1차 비교 기준은 다음과 같습니다.

- train loss와 validation loss 차이가 너무 벌어지지 않는지
- validation loss가 안정적으로 낮아지는지
- validation accuracy가 개선되는지
- best epoch가 너무 이른 시점에 끝나지 않는지
- 학습 시간이 과도하게 늘어나지 않는지

이 비교를 통해 가장 좋은 후보 모델을 선택한 뒤, 그 모델을 중심으로 심층 분석을 진행합니다.

심층 분석에서는 다음 항목을 확인합니다.

- test accuracy
- confusion matrix
- precision
- recall
- 오분류 이미지
- 오분류 원인 유형

이 순서를 선택한 이유는 다음과 같습니다.

- loss와 accuracy는 여러 실험을 빠르게 비교하기 좋습니다.
- confusion matrix, precision, recall은 모델이 어떤 방식으로 틀리는지 확인하는 데 좋습니다.
- 오분류 이미지는 숫자 지표만으로 알 수 없는 실패 원인을 직접 확인하는 데 좋습니다.
- 따라서 모든 실험에 대해 처음부터 심층 분석을 하기보다, 좋은 후보 모델을 고른 뒤 깊게 분석하는 것이 더 효율적입니다.

발표에서는 다음과 같이 설명할 수 있습니다.

> 여러 개선 실험을 먼저 loss와 accuracy 기준으로 비교했습니다. 그 후 가장 좋은 후보 모델을 선택해 confusion matrix, precision, recall, 오분류 이미지를 확인했습니다. accuracy만으로는 어떤 방향의 오류가 많은지 알 수 없기 때문에, 최종 후보 모델을 대상으로 심층 분석을 진행해 모델이 어떤 상황에서 틀리는지 원인을 파악하려고 했습니다.

## 오분류 분석 이후 개선 방향 메모

오분류 이미지를 직접 확인한 뒤, 다음 네 가지를 향후 개선 방향으로 정리합니다.

1. **head ROI 활용**
   - Oxford-IIIT Pet Dataset에는 동물의 머리 위치 정보가 포함되어 있습니다.
   - dog/cat 구분에는 얼굴, 귀, 눈, 코 주변 특징이 중요하므로, 전체 이미지보다 얼굴 중심 영역을 활용하면 성능 개선 가능성이 있습니다.

2. **segmentation 활용**
   - Oxford-IIIT Pet Dataset에는 동물과 배경을 나누는 segmentation annotation도 포함되어 있습니다.
   - 배경이 복잡한 이미지에서 모델이 배경을 잘못된 근거로 사용할 수 있으므로, 동물 영역에 더 집중하도록 만들 수 있습니다.

3. **실패 유형에 맞춘 데이터 증강**
   - OpenCV 증강을 무작정 많이 적용하는 것보다, 실제 오분류 원인에 맞춰 증강하는 방향이 더 적절합니다.
   - 예를 들어 어두운 사진에서 많이 틀리면 밝기/대비 증강을, 흐린 사진에서 많이 틀리면 blur 대응 증강을, 잘림이나 위치 변화에서 많이 틀리면 crop/rotate 계열 증강을 조정합니다.

4. **전이학습 적용**
   - ResNet, MobileNet, EfficientNet처럼 ImageNet에서 사전 학습된 모델을 활용할 수 있습니다.
   - 직접 만든 작은 CNN은 학습 과정을 설명하기 좋지만, 성능을 더 높이려면 사전 학습 모델의 feature extractor를 활용하는 것이 효과적일 수 있습니다.

발표에서는 다음과 같이 설명할 수 있습니다.

> 오분류 이미지를 직접 확인한 결과, 모델이 단순히 무작위로 틀린 것이 아니라 얼굴 일부만 보이는 경우, 배경이 복잡한 경우, 품종 특성이 애매한 경우처럼 반복되는 실패 유형이 있었습니다. 개선 방향으로는 첫째, head ROI를 활용해 얼굴 중심 특징을 더 잘 보도록 하고, 둘째, segmentation을 활용해 배경 영향을 줄이며, 셋째, 실패 유형에 맞춘 데이터 증강을 적용할 수 있습니다. 마지막으로 더 높은 성능을 목표로 한다면 ResNet이나 MobileNet 같은 사전 학습 모델을 활용한 전이학습을 고려할 수 있습니다.
