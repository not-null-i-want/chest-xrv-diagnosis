from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from univdt.components.base import BaseComponent
from univdt.utils.image import load_image

# 구동 함수들 임포트시켜주기
# Path 클래스  : 파일 시스템 경로를 다루는 데 사용
# Any  : 모든 유형을 나타내는 특수한 유형 힌트
# Optional : 특정 유형 또는 None을 나타냄

MAPPER = {'Normal': 0, # 정상
          'Effusion': 1, # 흉막삼출, 흉수 -> 흉막에서 체액 성분이 스며나오는 삼출 증상이 나타나는 질환
          'Emphysema': 2, # 폐 기종 -> 기관지나 폐에 염증이 생겨 폐의 공기 흐름이 막힌 것
          'Atelectasis': 3, # 무기폐 ->  폐의 일부가 팽창된 상태를 유지하지 못하고, 부피가 줄어 쭈그러든 것
          'Edema': 4, # 폐 부종 -> 혈관 밖의 폐조직인 폐간질 및 폐포에 비정상적으로 액체가 고이는 것 즉 폐에 물이 찬 것
          'Consolidation': 5, # 폐 경화 -> 액체나 세포등이 폐포의 공기를 대체하여 폐가 단단하게 된 상태
          'Pleural_thickening': 6, # 흉막비후 -> 흉막이 딱딱하게 굳어 두꺼워지는 증상
          'Hernia': 7, # 폐 탈장 -> 폐가 흉벽 밖으로 돌출되는 질환
          'Mass': 8, # 폐 종괴 ->  결절에 비해 큰 혹으로 3cm보다 큼
          'Nodule': 9, # 폐 결절 -> 폐 내부에 생긴 지름 3cm 미만의 작은 구상 병변
          'Pneumothorax': 10, # 기흉 -> 허파와 흉부벽 사이의 흉강에 공기가 비정상적으로 모이는 현상
          'Pneumonia': 11, # 폐렴 -> 세균이나 바이러스, 곰팡이 등의 미생물로 인한 감염으로 발생하는 폐의 염증
          'Fibrosis': 12, # 폐 섬유종 -> 폐 조직이 굳어서 심각한 호흡 장애를 불러일으키는 호흡기 질환
          'Infiltration': 13 # 침윤음영 -> 조직이나 세포에 정상이 아닌 물질이나 비정상적으로 과다한 물질이 축적, 침착하는 것, 또는 축적된 물질 
          }

# 데이터세트의 레이블 인덱스를 수동으로 정의해줌
# 데이터셋의 클래스 레이블을 숫자로 매핑하여 모델 훈련에 사용될 수 있음

class NIH(BaseComponent):
    """
    NIH Chest X-ray 14 dataset    
    Args:
        root : root folder for dataset
        split : 'train', 'val', 'test' and 'trainval'
        transform : Composed transforms

        root : 훈련/테스트 데이터가 저장되는 경로
        split : 훈련/검증/테스트 그리고 훈련 및 검증 데이터로 분할 
        transform : 여러 변환을 조합하여 만든 변환
    """

    AVAILABLE_KEYS = ['age', 'gender', 'view_position', 'patient_id', 'follow_up'] # 데이터셋에서 사용할 수 있는 특성(키) 리스트 

    def __init__(self, root_dir: str, split: str, transform=None,
                 additional_keys: list[str] | None = None): # __init__을 사용하여 데이터셋 클래스의 인스턴스를 초기화(받을 매개변수 : root_dir, split, transform, additional_keys)
        super().__init__(root_dir, split, transform, additional_keys) # BaseComponent(부모) 클래스를 상속받아서 부모 클래스의 초기화 변수를 호출함
        self.check_split(['train', 'val', 'trainval', 'test']) # 데이터셋 분할이 올바르게 이루어졌는지 확인하는 메서드를 호출함 올바르지 않을시 에러 발생

        if self.additional_keys:
            assert all([key in self.AVAILABLE_KEYS for key in self.additional_keys]), \
                f'Invalid additional keys: {self.additional_keys}' 
    
        # 만약 additional key가 비어있지 않다면, available_keys에 포함된 특성들 중 어느 것에도 속하지 않는지 확인하고, 속하지 않을 시 에러 생성

        self.raw_data = self._load_paths() # 데이터 경로를 로드하여 raw data에 저장


    def __getitem__(self, index: int) -> dict[str, Any]: # 데이터셋 객체를 인덱싱할 때 호출되는 getitem 메서드를 정의 후 데이터에 대한 정보를 딕셔너리로 포장하여 반환
        data = self.load_data(index) # 주어진 인덱스에 해당하는 데이터를 로드
        image: np.ndarray = data['image'] # 이미지를 넘파이 배열로 추출
        label: np.ndarray = data['label'] # 라벨을 넘파이 배열로 추출
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']

        # 데이터셋이 가진 변환이 None이 아닌 경우, 이미지에 해당 반환을 적용함
        # 입력받은 이미지를 딕셔너리 형태로 반환되는 형식으로 이미지에 변환을 적용하고 변환된 이미지를 추출
        # convert image to pytorch tensor(이미지를 파이토치 형식으로 변환)
        image = image.transpose(2, 0, 1) # 이미지의 차원을 변환하는 연산(채널(C), 높이(H), 너비(W))
        image = image.astype('float32') / 255.0 # 이미지를 0과 1 사이로 정규화
        image = torch.Tensor(image) # pytorch 텐서로 변환
        output = {'image': image, 'label': label,
                  'path': data['path']} # 이미지, 레이블, 데이터셋의 정보, 이미지의 경로를 포함한 반환할 출력 딕셔너리를 생성
        output.update({key: data[key] for key in self.additional_keys}) # 추가적인 키들과 해당 값들을 출력 딕셔너리에 추가
        return output # 완성된 출력 딕셔너리를 반환
    


    def __len__(self) -> int:  # 데이터셋의 총 길이를 반환
        return len(self.raw_data)

    def load_data(self, index: int) -> dict[str, Any]: # 인덱스에 해당하는 데이터를 로드한 후 해당 데이터를 딕셔너리 형태로 반환
        raw_data = self.raw_data[index] # self.raw_data에서 해당 인덱스에 해당하는 데이터를 가져옴
        # load image
        image_path = Path(self.root_dir) / raw_data['path'] # 이미지의 경로를 가져와서 해당 경로의 이미지 파일을 읽어옵니다
        assert image_path.exists(), f'Image {image_path} does not exist' # 이미지 파일이 실제로 존재하는지 확인하는데 사용
        image = load_image(image_path, out_channels=1)  # normalized to [0, 255] 
        # -> 가져온 이미지를 1채널 그레이스케일로 로드, 이미지는 0에서 255 사이의 값으로 정규화
        label = raw_data['findings'] # raw_data 딕셔너리의 'findings' 키에 저장되어 있는 이미지에 대한 레이블 정보를 가져옴
        label = np.array(label, dtype=np.int64) # 가져온 레이블 정보를 NumPy 배열로 변환, 데이터 타입은 64비트 정수타입

        # load etc data -> 기타 데이터(나이, 성별, 조사 위치, 환자 ID, 후속 조사)를 가져와서 변수에 할당
        age = raw_data['age']
        gender = raw_data['gender']
        view_position = raw_data['view-position']
        patient_id = raw_data['pid']
        follow_up = raw_data['follow-up']

        # 로드한 데이터 및 정보를 딕셔너리 형태로 구성하여 반환(이미지, 레이블, 이미지 경로, 나이, 성별, 조사 위치, 환자 ID, 후속 조사 등의 키-값이 포함)
        return {'image': image, 'label': label, 'path': str(image_path),
                'age': age, 'gender': gender, 'view_position': view_position,
                'pid': patient_id, 'fup': follow_up}

    def _load_paths(self): # 데이터셋의 이미지 경로와 해당 이미지의 레이블을 로드하는 함수
        import pandas as pd # 판다스 라이브러리 임포트
        # path, split, findings, age, gender, view-position, pid, follow-up
        df = pd.read_csv(self.root_dir / 'nih.csv') # nih.csv파일 데이터프레임으로 변환
        df = df[df['split'].isin([self.split])] if self.split != 'trainval' \
            else df[df['split'].isin(['train', 'val'])]
        # 데이터셋의 `split` 열 값에 따라 데이터를 필터링 -> 만약 `split`이 'trainval'이 아니면 'train' 또는 'val'인 경우에만 해당 데이터를 선택
        for i, row in df.iterrows(): # 후처리
            # string list to list in findings(각 행을 반복하면서 `findings` 열의 값을 리스트로 변환)
            findings = str(row['findings'])
            findings = findings.strip('[]').split(',')
            findings = [int(f.strip()) for f in findings]
            df.at[i, 'findings'] = findings
        # add 'images/' to path
        df['path'] = df['path'].apply(lambda x: 'images/' + x) # 이미지의 경로를 수정하여 'images/'를 경로의 접두사로 추가
        return [dict(row) for _, row in df.iterrows()] # 각 행을 딕셔너리로 변환하고, 이를 리스트에 저장하여 반환
