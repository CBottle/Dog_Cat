# 02-1 OpenCV Augmentation Training Design

## Goal

`02_1_opencv_augmentation_training.ipynb`는 `02_training_baseline.ipynb`에서 만든 baseline 학습을 확장하는 노트북입니다.

목표는 단순히 점수를 높이는 것이 아니라, 발표와 결과 문서에서 `WHY -> HOW -> RESULT` 흐름으로 접근 과정을 설명할 수 있게 만드는 것입니다.

## WHY

실제 강아지/고양이 사진은 조명, 대비, 흔들림, 회전, 좌우 방향, 배경이 다양합니다.

원본 train 데이터만 학습하면 이런 조건 변화에 약할 수 있으므로, train 이미지에 OpenCV 증강을 적용해 모델이 더 다양한 조건을 경험하도록 합니다.

## HOW

- `01_data_split.ipynb`에서 만든 70/15/15 split CSV를 재사용합니다.
- split은 다시 하지 않습니다.
- train set에만 OpenCV 증강을 적용합니다.
- validation/test set은 원본 그대로 둡니다.
- baseline과 같은 SimpleCNN 구조를 사용합니다.
- baseline과 같은 image size, batch size, loss function, optimizer 계열을 사용합니다.
- best model checkpoint와 EarlyStopping은 `02`의 개선 학습 루프와 같은 방식으로 사용합니다.

## RESULT

노트북은 다음 결과를 남깁니다.

- 증강 이미지 샘플 확인
- 원본 train 개수와 증강 후 train 개수 비교
- OpenCV 증강 모델 loss curve
- OpenCV 증강 모델 best checkpoint
- baseline과 비교하기 위한 요약 기록

## Notebook Structure

1. Step 0. 이번 노트북을 따로 만드는 이유를 정리합니다.
2. Step 1. WHY: 왜 OpenCV 증강을 하는지 정리합니다.
3. Step 2. HOW: 비교 실험 조건을 고정합니다.
4. Step 3. 필요한 패키지를 불러옵니다.
5. Step 4. 프로젝트 경로와 저장 위치를 정합니다.
6. Step 5. 01의 split CSV를 불러옵니다.
7. Step 6. OpenCV 증강 함수를 만듭니다.
8. Step 7. 증강 이미지 샘플을 확인합니다.
9. Step 8. train 증강 이미지 파일을 생성합니다.
10. Step 9. 원본 train과 증강 train을 하나의 DataFrame으로 합칩니다.
11. Step 10. Dataset/DataLoader를 만듭니다.
12. Step 11. baseline과 같은 SimpleCNN을 사용합니다.
13. Step 12. 개선 학습 루프로 증강 모델을 학습합니다.
14. Step 13. RESULT: loss curve와 학습 기록을 저장합니다.
15. Step 14. baseline과 비교할 항목을 정리합니다.
16. Step 15. 발표용 WHY/HOW/RESULT 문장 초안을 정리합니다.

## Design Decisions

- 증강은 offline 방식으로 실제 이미지 파일을 저장합니다. 발표에서 증강된 이미지와 데이터 수 증가를 설명하기 쉽기 때문입니다.
- 증강 이미지는 `data/processed/augmented/opencv_train/`에 저장합니다.
- 증강 metadata CSV는 `data/processed/pet_binary_split_70_15_15_opencv_augmented_train.csv`로 저장합니다.
- checkpoint는 `outputs/checkpoints/best_opencv_augmented_simplecnn.pt`로 저장합니다.
- 생성 데이터와 checkpoint는 Git에 커밋하지 않습니다.

## Comparison Principle

공정한 비교를 위해 baseline과 augmentation 실험의 차이는 `train 데이터 증강 여부`만으로 제한합니다.

- 같은 split CSV를 사용합니다.
- 같은 validation/test를 사용합니다.
- 같은 image size와 batch size를 사용합니다.
- 같은 SimpleCNN 구조를 사용합니다.
- 같은 loss function과 optimizer 계열을 사용합니다.
