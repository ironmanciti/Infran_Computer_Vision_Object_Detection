# SSD 모델을 사용한 객체 탐지 미세조정

이 프로젝트는 PyTorch의 SSD(Single Shot MultiBox Detector) 모델을 사용하여 사용자 정의 데이터셋으로 객체 탐지 모델을 미세조정하는 코드입니다.

## 주요 기능

- **VIA 형식 어노테이션을 YOLO 형식으로 변환**
- **SSD 모델 미세조정**
- **사용자 정의 데이터셋 지원**
- **예측 결과 시각화**

## 파일 구조

```
├── convert_via_to_yolo.py    # VIA 형식을 YOLO 형식으로 변환
├── ssd_finetuning.py         # SSD 모델 미세조정 메인 코드
├── balloon/                  # 원본 balloon 데이터셋 (VIA 형식)
│   ├── train/
│   │   ├── via_region_data.json
│   │   └── *.jpg
│   └── val/
│       ├── via_region_data.json
│       └── *.jpg
└── balloon_yolo/             # 변환된 YOLO 형식 데이터셋
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── data.yaml
```

## 설치 및 실행

### 1. 필요한 라이브러리 설치

```bash
pip install torch torchvision matplotlib pillow scikit-learn pyyaml
```

### 2. 데이터 변환 (VIA → YOLO)

VIA 형식의 어노테이션을 YOLO 형식으로 변환합니다:

```bash
python convert_via_to_yolo.py
```

### 3. SSD 모델 미세조정

```bash
python ssd_finetuning.py
```

## 코드 설명

### convert_via_to_yolo.py

VIA 형식의 JSON 어노테이션 파일을 YOLO 형식의 텍스트 파일로 변환합니다.

**주요 함수:**
- `convert_via_to_yolo()`: VIA JSON을 YOLO 형식으로 변환
- `create_yaml_config()`: YOLO 설정 파일 생성

### ssd_finetuning.py

SSD 모델을 사용한 객체 탐지 미세조정을 수행합니다.

**주요 클래스:**

#### CustomObjectDetectionDataset
- 사용자 정의 객체 탐지 데이터셋 클래스
- YOLO 형식 어노테이션 파일 로드
- 이미지와 바운딩 박스 정보 처리

#### SSDFineTuning
- SSD 모델 미세조정을 위한 메인 클래스
- 데이터 준비, 모델 훈련, 예측 기능 제공

**주요 메서드:**
- `prepare_data()`: 데이터셋 준비 (VIA/YOLO 형식 자동 감지)
- `train()`: 모델 훈련
- `predict()`: 예측 수행
- `save_model()` / `load_model()`: 모델 저장/로드

## 설정 옵션

### 클래스 설정
```python
class_names = ["balloon"]  # 탐지할 클래스 이름
num_classes = len(class_names) + 1  # background 포함
```

### 훈련 설정
```python
epochs = 5              # 훈련 에폭 수
learning_rate = 0.001   # 학습률
batch_size = 2          # 배치 크기
confidence_threshold = 0.5  # 예측 신뢰도 임계값
```

## 사용 예시

### 1. 기본 실행
```python
# 메인 함수 실행
python ssd_finetuning.py
```

### 2. 커스텀 설정으로 실행
```python
from ssd_finetuning import SSDFineTuning

# SSD 미세조정 객체 생성
ssd_trainer = SSDFineTuning(num_classes=2, device='cuda')

# 데이터 준비
train_imgs, val_imgs, train_anns, val_anns = ssd_trainer.prepare_data("balloon_yolo")

# 데이터 로더 생성
train_loader, val_loader = ssd_trainer.create_data_loaders(
    train_imgs, val_imgs, train_anns, val_anns, 
    class_names=["balloon"], batch_size=4
)

# 모델 훈련
ssd_trainer.train(train_loader, val_loader, epochs=10, learning_rate=0.001)

# 모델 저장
ssd_trainer.save_model("my_ssd_model.pth")
```

## 출력 파일

- `ssd_finetuned_model.pth`: 훈련된 모델 가중치
- `ssd_prediction_result.jpg`: 예측 결과 시각화 이미지

## 주의사항

1. **메모리 사용량**: SSD 모델은 메모리를 많이 사용하므로 GPU 메모리가 충분한지 확인하세요.
2. **데이터 형식**: 어노테이션 파일이 YOLO 형식이어야 합니다.
3. **클래스 수**: 새로운 클래스를 추가할 때는 `num_classes`를 올바르게 설정하세요.

## 문제 해결

### 메모리 부족 오류
- `batch_size`를 줄여보세요 (예: 2 → 1)
- 이미지 크기를 줄여보세요 (예: 300 → 224)

### 데이터 로딩 오류
- 어노테이션 파일 경로가 올바른지 확인하세요
- 이미지와 어노테이션 파일의 이름이 일치하는지 확인하세요

### 훈련이 수렴하지 않는 경우
- `learning_rate`를 낮춰보세요
- `epochs`를 늘려보세요
- 데이터 품질을 확인하세요

## 참고 자료

- [PyTorch SSD Documentation](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)
- [YOLO Format Specification](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
- [VIA Annotation Tool](http://www.robots.ox.ac.uk/~vgg/software/via/) 