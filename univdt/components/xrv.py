import random
import cv2
import numpy as np
from functools import partial
from univdt.components import NIH
import torch
import torchvision.transforms as transforms
import xrv

def generate_random_index(max_index):
    generated_indexes = set()  # 생성된 인덱스를 추적하기 위한 빈 집합 생성
    while True:
        random_index = random.randint(0, max_index - 1)
        if random_index not in generated_indexes:
            generated_indexes.add(random_index)
            return random_index  # 생성된 랜덤 인덱스 반환

def load_data(root_dir, split, index):
    nih = partial(NIH, root_dir=root_dir, transform=None)
    dataset = nih(split=split)
    return dataset.load_data(index)  # 데이터셋에서 해당 인덱스의 데이터 로드 후 반환

def resize_image(image, target_size):
    image_resized = cv2.resize(image, target_size)  # 이미지 크기를 타겟 사이즈로 조정
    image_resized = np.expand_dims(image_resized, axis=2)  # 차원 확장
    return image_resized  # 크기 조정된 이미지 반환

def normalize_image(image):
    image_normalized = xrv.datasets.normalize(image, 255)  # 이미지 정규화
    image_normalized = image_normalized.mean(2)[None, ...]  # 차원 확장
    image_normalized = image_normalized.astype(dtype=np.float32)  # 데이터 타입 변환
    return image_normalized  # 정규화된 이미지 반환

def predict_disease(image, model):
    outputs = model(image[None, ...])  # 이미지를 모델에 입력하여 질병 예측
    return outputs  # 질병 예측 결과 반환

def process_image(image, label):
    def resize_image(image, size):
        resized_image = cv2.resize(image, size)  # 이미지 크기 조정
        return resized_image  # 크기 조정된 이미지 반환

    def normalize_image(image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        normalized_image = transform(image)  # 이미지 정규화
        return normalized_image  # 정규화된 이미지 반환

    image_resized = resize_image(image, (512, 512))  # 이미지 크기 조정
    print(image_resized.shape, label)  # 이미지 크기 조정 후 출력

    image_normalized = normalize_image(image_resized)  # 이미지 정규화
    print(image_normalized.shape, label)  # 이미지 정규화 후 출력
    image_normalized = torch.from_numpy(image_normalized)  # 넘파이 배열을 텐서로 변환
    print(image_normalized.shape, label, image_normalized.dtype)  # 텐서로 변환 후 출력

    model = xrv.models.ResNet(weights="resnet50-res512-all")  # 사전 학습된 ResNet 모델 로드
    outputs = predict_disease(image_normalized, model)  # 질병 예측

    return outputs  # 질병 예측 결과 반환

def get_disease_title(label, MAPPER):
    if isinstance(label, np.ndarray):
        diseases = []
        for l in label:
            l = int(l)  # label을 정수로 변환
            if l not in MAPPER.values():
                print(f"Label {l} not found in MAPPER.")
            else:
                disease_name = list(MAPPER.keys())[list(MAPPER.values()).index(l)]  # 질병 이름 조회
                diseases.append(disease_name)
        unique_diseases = set(diseases)  # 중복 제거
        title = f'Disease: {", ".join(unique_diseases)}'  # 질병 제목 생성

    else:
        label = int(label)  # label을 정수로 변환
        if label not in MAPPER.values():
            print(f"Label {label} not found in MAPPER.")
            title = "Unknown Disease"
        else:
            disease_name = list(MAPPER.keys())[list(MAPPER.values()).index(label)]  # 질병 이름 조회
            title = f"Disease: {disease_name}"  # 질병 제목 생성

    return title  # 질병 제목 반환

def get_disease_probabilities(diseases, output_probabilities_dict):
    probabilities = []
    
    for disease in diseases:
        if disease in output_probabilities_dict:
            probability = output_probabilities_dict[disease]  # 질병 확률 조회
            probability_percent = round(probability * 100, 1)  # 백분율로 변환
            probabilities.append(f"질병: {disease}, 확률: {probability_percent:.2f}%")  # 질병과 확률 정보 추가
    
    return probabilities  # 질병 확률 리스트 반환
