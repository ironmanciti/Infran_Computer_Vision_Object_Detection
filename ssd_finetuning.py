# -*- coding: utf-8 -*-
"""
SSD 모델을 사용한 사용자 정의 데이터셋 미세조정
- Single Shot MultiBox Detector를 사용한 객체 탐지 모델 미세조정
"""

import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import json
import glob
from sklearn.model_selection import train_test_split
import yaml
import shutil

# COCO 클래스 이름 (기본 80개 클래스 + background)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class CustomObjectDetectionDataset(Dataset):
    """사용자 정의 객체 탐지 데이터셋 클래스"""
    
    def __init__(self, image_paths, annotation_paths, transform=None, class_names=None):
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.transform = transform
        self.class_names = class_names or ['object']  # 기본 클래스 이름
        
        # 클래스 이름을 인덱스로 매핑
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names, start=1)}
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 이미지 로드
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # 어노테이션 로드 (YOLO 형식 또는 COCO 형식)
        annotation_path = self.annotation_paths[idx]
        boxes, labels = self.load_annotations(annotation_path, image.size)
        
        # 텐서 변환
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
        }
        
        if self.transform:
            image, target = self.transform(image, target)
        
        return image, target
    
    def load_annotations(self, annotation_path, image_size):
        """어노테이션 파일 로드 (YOLO 형식 지원)"""
        boxes = []
        labels = []
        
        if annotation_path.endswith('.txt'):  # YOLO 형식
            with open(annotation_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # YOLO 좌표를 COCO 좌표로 변환
                        x_min = (x_center - width/2) * image_size[0]
                        y_min = (y_center - height/2) * image_size[1]
                        x_max = (x_center + width/2) * image_size[0]
                        y_max = (y_center + height/2) * image_size[1]
                        
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(class_id + 1)  # background가 0이므로 +1
        
        return boxes, labels

class SSDFineTuning:
    """SSD 모델 미세조정 클래스"""
    
    def __init__(self, num_classes, device='cuda'):
        self.device = device
        self.num_classes = num_classes
        
        # 사전 훈련된 SSD 모델 로드
        self.model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
        
        # 분류기 헤드를 새로운 클래스 수에 맞게 수정
        in_channels = 256  # SSD의 분류기 입력 채널 수
        num_anchors = 6    # SSD의 앵커 박스 수
        
        # 분류기와 박스 예측기 수정
        self.model.head.classification_head.cls_logits = torch.nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, padding=1
        )
        
        self.model.to(device)
        
    def prepare_data(self, dataset_path, train_ratio=0.8):
        """데이터셋 준비"""
        # YOLO 형식 데이터셋 구조 확인
        train_images_dir = os.path.join(dataset_path, "train", "images")
        train_labels_dir = os.path.join(dataset_path, "train", "labels")
        val_images_dir = os.path.join(dataset_path, "val", "images")
        val_labels_dir = os.path.join(dataset_path, "val", "labels")
        
        # 기존 VIA 형식 데이터셋 구조 확인
        if not os.path.exists(train_images_dir):
            # VIA 형식 데이터셋인 경우
            return self.prepare_via_data(dataset_path, train_ratio)
        
        # YOLO 형식 데이터셋인 경우
        return self.prepare_yolo_data(dataset_path)
    
    def prepare_yolo_data(self, dataset_path):
        """YOLO 형식 데이터셋 준비"""
        train_images_dir = os.path.join(dataset_path, "train", "images")
        train_labels_dir = os.path.join(dataset_path, "train", "labels")
        val_images_dir = os.path.join(dataset_path, "val", "images")
        val_labels_dir = os.path.join(dataset_path, "val", "labels")
        
        # 학습 이미지와 어노테이션 파일 찾기
        train_images = []
        train_annotations = []
        
        if os.path.exists(train_images_dir):
            for img_file in glob.glob(os.path.join(train_images_dir, "*.jpg")):
                base_name = os.path.splitext(os.path.basename(img_file))[0]
                ann_file = os.path.join(train_labels_dir, f"{base_name}.txt")
                if os.path.exists(ann_file):
                    train_images.append(img_file)
                    train_annotations.append(ann_file)
        
        # 검증 이미지와 어노테이션 파일 찾기
        val_images = []
        val_annotations = []
        
        if os.path.exists(val_images_dir):
            for img_file in glob.glob(os.path.join(val_images_dir, "*.jpg")):
                base_name = os.path.splitext(os.path.basename(img_file))[0]
                ann_file = os.path.join(val_labels_dir, f"{base_name}.txt")
                if os.path.exists(ann_file):
                    val_images.append(img_file)
                    val_annotations.append(ann_file)
        
        return train_images, val_images, train_annotations, val_annotations
    
    def prepare_via_data(self, dataset_path, train_ratio):
        """VIA 형식 데이터셋 준비 (기존 방식)"""
        # 이미지 파일 찾기
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(dataset_path, 'images', ext)))
            image_files.extend(glob.glob(os.path.join(dataset_path, '**', ext)))
        
        # 어노테이션 파일 찾기
        annotation_files = []
        for img_path in image_files:
            # YOLO 형식 어노테이션 파일 경로 생성
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            annotation_path = os.path.join(dataset_path, 'labels', f'{base_name}.txt')
            if os.path.exists(annotation_path):
                annotation_files.append(annotation_path)
            else:
                # 다른 위치에서 찾기
                for root, dirs, files in os.walk(dataset_path):
                    if f'{base_name}.txt' in files:
                        annotation_files.append(os.path.join(root, f'{base_name}.txt'))
                        break
                else:
                    annotation_files.append(None)
        
        # 유효한 이미지-어노테이션 쌍만 필터링
        valid_pairs = [(img, ann) for img, ann in zip(image_files, annotation_files) if ann is not None]
        
        if not valid_pairs:
            raise ValueError("유효한 이미지-어노테이션 쌍을 찾을 수 없습니다.")
        
        image_files, annotation_files = zip(*valid_pairs)
        
        # 학습/검증 분할
        train_imgs, val_imgs, train_anns, val_anns = train_test_split(
            image_files, annotation_files, test_size=1-train_ratio, random_state=42
        )
        
        return train_imgs, val_imgs, train_anns, val_anns
    
    def create_data_loaders(self, train_imgs, val_imgs, train_anns, val_anns, class_names, batch_size=4):
        """데이터 로더 생성"""
        # 변환 정의
        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
        ])
        
        # 데이터셋 생성
        train_dataset = CustomObjectDetectionDataset(
            train_imgs, train_anns, transform=transform, class_names=class_names
        )
        val_dataset = CustomObjectDetectionDataset(
            val_imgs, val_anns, transform=transform, class_names=class_names
        )
        
        # 데이터 로더 생성
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        
        return train_loader, val_loader
    
    def collate_fn(self, batch):
        """배치 데이터 정렬 함수"""
        return tuple(zip(*batch))
    
    def train(self, train_loader, val_loader, epochs=10, learning_rate=0.001):
        """모델 훈련"""
        # 옵티마이저 설정
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 학습 루프
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                optimizer.zero_grad()
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                losses.backward()
                optimizer.step()
                
                total_loss += losses.item()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {losses.item():.4f}')
            
            # 검증
            val_loss = self.validate(val_loader)
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')
    
    def validate(self, val_loader):
        """검증"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
        
        return total_loss / len(val_loader)
    
    def predict(self, image_path, confidence_threshold=0.5):
        """예측 수행"""
        self.model.eval()
        
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
        ])
        
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        # 예측
        with torch.no_grad():
            predictions = self.model(input_batch)[0]
        
        # 결과 필터링
        boxes = predictions['boxes']
        labels = predictions['labels']
        scores = predictions['scores']
        
        # 신뢰도 임계값 적용
        mask = scores > confidence_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        
        return boxes, labels, scores
    
    def save_model(self, path):
        """모델 저장"""
        torch.save(self.model.state_dict(), path)
        print(f"모델이 {path}에 저장되었습니다.")
    
    def load_model(self, path):
        """모델 로드"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"모델이 {path}에서 로드되었습니다.")

def visualize_predictions(image_path, boxes, labels, scores, class_names, save_path=None):
    """예측 결과 시각화"""
    image = Image.open(image_path)
    
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    
    for box, label, score in zip(boxes, labels, scores):
        x_min, y_min, x_max, y_max = box.cpu().numpy()
        width = x_max - x_min
        height = y_max - y_min
        
        # 박스 그리기
        rect = patches.Rectangle((x_min, y_min), width, height, 
                               linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # 클래스 이름과 점수 표시
        class_name = class_names[label.item()] if label.item() < len(class_names) else f'class_{label.item()}'
        text = f"{class_name}: {score:.2f}"
        ax.text(x_min, y_min-5, text, fontsize=12, color='red',
                bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"결과가 {save_path}에 저장되었습니다.")
    
    plt.show()

def main():
    """메인 실행 함수"""
    # 설정
    dataset_path = "balloon_yolo"  # 변환된 YOLO 형식 데이터셋 경로
    class_names = ["balloon"]  # 클래스 이름 (background 제외)
    num_classes = len(class_names) + 1  # background 포함
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"사용 디바이스: {device}")
    print(f"클래스 수: {num_classes}")
    print(f"클래스 이름: {class_names}")
    
    # 데이터셋이 변환되지 않은 경우 변환 실행
    if not os.path.exists(dataset_path):
        print("YOLO 형식 데이터셋이 없습니다. VIA 형식을 YOLO 형식으로 변환합니다...")
        os.system("python convert_via_to_yolo.py")
    
    # SSD 미세조정 객체 생성
    ssd_trainer = SSDFineTuning(num_classes=num_classes, device=device)
    
    # 데이터 준비
    print("데이터셋 준비 중...")
    train_imgs, val_imgs, train_anns, val_anns = ssd_trainer.prepare_data(dataset_path)
    
    print(f"학습 이미지 수: {len(train_imgs)}")
    print(f"검증 이미지 수: {len(val_imgs)}")
    
    if len(train_imgs) == 0:
        print("학습할 이미지가 없습니다. 데이터셋 경로를 확인해주세요.")
        return
    
    # 데이터 로더 생성
    train_loader, val_loader = ssd_trainer.create_data_loaders(
        train_imgs, val_imgs, train_anns, val_anns, class_names, batch_size=2
    )
    
    # 모델 훈련
    print("모델 훈련 시작...")
    ssd_trainer.train(train_loader, val_loader, epochs=5, learning_rate=0.001)
    
    # 모델 저장
    ssd_trainer.save_model("ssd_finetuned_model.pth")
    
    # 테스트 예측
    print("테스트 예측 수행...")
    
    # 훈련된 모델로 테스트 이미지 예측
    if len(train_imgs) > 0:
        test_image_path = train_imgs[0]  # 첫 번째 훈련 이미지로 테스트
        boxes, labels, scores = ssd_trainer.predict(test_image_path, confidence_threshold=0.5)
        
        # 결과 시각화
        all_class_names = ['__background__'] + class_names
        visualize_predictions(test_image_path, boxes, labels, scores, all_class_names, 
                            save_path="ssd_prediction_result.jpg")
    else:
        print("테스트할 이미지가 없습니다.")

if __name__ == "__main__":
    main() 