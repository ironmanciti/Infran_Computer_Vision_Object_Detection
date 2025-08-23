# -*- coding: utf-8 -*-
"""
SSD 미세조정 실행 스크립트
간단한 설정으로 SSD 모델 미세조정을 실행할 수 있습니다.
"""

import os
import sys
from ssd_finetuning import SSDFineTuning, visualize_predictions

def main():
    """SSD 미세조정 실행"""
    
    print("=" * 50)
    print("SSD 모델 미세조정 시작")
    print("=" * 50)
    
    # 설정
    dataset_path = "balloon_yolo"  # 데이터셋 경로
    class_names = ["balloon"]      # 클래스 이름
    num_classes = len(class_names) + 1  # background 포함
    
    # GPU 사용 가능 여부 확인
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"사용 디바이스: {device}")
    print(f"클래스 수: {num_classes}")
    print(f"클래스 이름: {class_names}")
    print(f"데이터셋 경로: {dataset_path}")
    
    # 데이터셋이 변환되지 않은 경우 변환 실행
    if not os.path.exists(dataset_path):
        print("\nYOLO 형식 데이터셋이 없습니다. VIA 형식을 YOLO 형식으로 변환합니다...")
        try:
            os.system("python convert_via_to_yolo.py")
            print("데이터 변환이 완료되었습니다.")
        except Exception as e:
            print(f"데이터 변환 중 오류 발생: {e}")
            return
    
    # SSD 미세조정 객체 생성
    print("\nSSD 모델 초기화 중...")
    ssd_trainer = SSDFineTuning(num_classes=num_classes, device=device)
    
    # 데이터 준비
    print("데이터셋 준비 중...")
    try:
        train_imgs, val_imgs, train_anns, val_anns = ssd_trainer.prepare_data(dataset_path)
        
        print(f"학습 이미지 수: {len(train_imgs)}")
        print(f"검증 이미지 수: {len(val_imgs)}")
        
        if len(train_imgs) == 0:
            print("학습할 이미지가 없습니다. 데이터셋 경로를 확인해주세요.")
            return
            
    except Exception as e:
        print(f"데이터 준비 중 오류 발생: {e}")
        return
    
    # 데이터 로더 생성
    print("데이터 로더 생성 중...")
    try:
        train_loader, val_loader = ssd_trainer.create_data_loaders(
            train_imgs, val_imgs, train_anns, val_anns, class_names, batch_size=2
        )
    except Exception as e:
        print(f"데이터 로더 생성 중 오류 발생: {e}")
        return
    
    # 모델 훈련
    print("\n모델 훈련 시작...")
    print("=" * 30)
    
    try:
        ssd_trainer.train(train_loader, val_loader, epochs=5, learning_rate=0.001)
        print("=" * 30)
        print("모델 훈련이 완료되었습니다!")
    except Exception as e:
        print(f"모델 훈련 중 오류 발생: {e}")
        return
    
    # 모델 저장
    print("\n모델 저장 중...")
    try:
        ssd_trainer.save_model("ssd_finetuned_model.pth")
    except Exception as e:
        print(f"모델 저장 중 오류 발생: {e}")
    
    # 테스트 예측
    print("\n테스트 예측 수행...")
    try:
        if len(train_imgs) > 0:
            test_image_path = train_imgs[0]  # 첫 번째 훈련 이미지로 테스트
            print(f"테스트 이미지: {os.path.basename(test_image_path)}")
            
            boxes, labels, scores = ssd_trainer.predict(test_image_path, confidence_threshold=0.5)
            
            print(f"탐지된 객체 수: {len(boxes)}")
            
            # 결과 시각화
            all_class_names = ['__background__'] + class_names
            visualize_predictions(test_image_path, boxes, labels, scores, all_class_names, 
                                save_path="ssd_prediction_result.jpg")
        else:
            print("테스트할 이미지가 없습니다.")
    except Exception as e:
        print(f"테스트 예측 중 오류 발생: {e}")
    
    print("\n" + "=" * 50)
    print("SSD 미세조정 완료!")
    print("=" * 50)
    print("출력 파일:")
    print("- ssd_finetuned_model.pth: 훈련된 모델")
    print("- ssd_prediction_result.jpg: 예측 결과")

if __name__ == "__main__":
    main() 