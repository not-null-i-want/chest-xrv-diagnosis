import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import copy
from pathlib import Path
from typing import Tuple, Union

import boto3
import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torchxrayvision as xrv
from flask import Flask, jsonify, request
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

app = Flask(__name__)

# AWS 설정
bucket_name = 'bucket-name'
AWS_ACCESS_KEY_ID = 'access-key'
AWS_SECRET_ACCESS_KEY = 'secret-access-key'

aws_region = 'region'

s3 = boto3.client('s3',
                  region_name=aws_region,
                  aws_access_key_id=AWS_ACCESS_KEY_ID,
                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

# HYPERPARAMETERS
_VERBOSE = True  # optional, set to True to enable print statements
_APPLY_TTA = True  # optional, set to True to enable test-time augmentation

#_MODEL = xrv.models.DenseNet(weights="densenet121-res224-nih").eval()
#_CAM = GradCAM(model=_MODEL, target_layers=_MODEL.features[-2][-1])
_MODEL = xrv.models.ResNet(weights="resnet50-res512-all").eval()
_CAM = GradCAM(model=_MODEL, target_layers=_MODEL.model.layer4)
_SEG_MODEL = xrv.baseline_models.chestx_det.PSPNet().eval()

_LESION_TO_CAMID = {lesion.lower(): i for i, lesion in enumerate(_MODEL.pathologies)}
_TARGET_LESIONS = ['atelectasis', 'consolidation', 'edema', 'effusion', 'emphysema',
                   'fibrosis', 'hernia', 'infiltration', 'mass', 'nodule',
                   'pleural_thickening', 'pneumonia', 'pneumothorax']

# 이미지 다운로드 폴더 생성
DOWNLOADS_FOLDER = 'downloads'
if not os.path.exists(DOWNLOADS_FOLDER):
    os.makedirs(DOWNLOADS_FOLDER)

def upload_file_to_s3(image_path, bucket_name, s3_file_name):
    s3.upload_file(image_path, bucket_name, s3_file_name)
    print(f"Uploaded image to S3: {s3_file_name}")
    file_url = f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{s3_file_name}"
    return file_url

def download_image_from_url(url: str, file_name: str) -> bool:
    """
    Download an image from the given URL and save it to a local file.

    Args:
        url (str): The URL of the image to download.
        file_name (str): The name to save the downloaded file as.

    Returns:
        bool: True if the download is successful, False otherwise.
    """
    try:
        response = requests.get(url)
        with open(file_name, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded image from URL: {url}")
        return True, file_name
    except Exception as e:
        print(f"Failed to download the image from URL: {url}\nError: {e}")
        return False, None


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    # check shape and add channel dimension if necessary
    image = np.expand_dims(image, axis=-1) if image.ndim == 2 else image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[-1] == 3 else image
    return image


def resize_images(image: np.ndarray, sizes: int | list[int]) -> list[np.ndarray]:
    """
    Resize the given image to the given sizes.

    Args:
        image (np.ndarray): The image to resize.
        sizes (int | list[int]): The sizes to resize the image to.
    Returns:
        list[np.ndarray]: A list of resized images.
    """
    if isinstance(sizes, int):
        sizes = [sizes]
    images = [cv2.resize(image, (size, size)) for size in sizes]
    return [np.expand_dims(img, axis=-1) for img in images]


def normailize_image(image: np.ndarray) -> torch.Tensor:
    """
    Normalize the given image.

    Args:
        image (np.ndarray): The image to normalize.

    Returns:
        np.ndarray: The normalized image.
    """
    assert image.ndim == 2 or image.shape[-1] == 1, "Image must be grayscale."
    image = xrv.datasets.normalize(image, 255)
    image = image.mean(2)[None, ...]
    image = image.astype(dtype=np.float32)
    image = torch.from_numpy(image)[None, ...]
    return image


@torch.no_grad()
def inference_images(images: list[torch.Tensor], model: torch.nn.Module) -> np.ndarray:
    """
    Perform inference on the given images using the given model.

    Args:
        images (list[torch.Tensor]): A list of images to perform inference on.
        model (torch.nn.Module): The model to use for inference.

    Returns:
        list[np.ndarray]: A list of probabilities for each image.
    """
    model = model.eval()
    concatenated_tensors = []
    unique_shapes = set(tensor.shape for tensor in images)
    for shape in unique_shapes:
        tensors_with_shape = [tensor for tensor in images if tensor.shape == shape]
        concatenated_tensor = torch.cat(tensors_with_shape, dim=0)
        concatenated_tensors.append(concatenated_tensor)
    outputs: list[torch.Tensor] = []
    for tensor in concatenated_tensors:
        output: torch.Tensor = model(tensor)
        output = output.detach().cpu()
        outputs.append(output)
    results = torch.cat(outputs, dim=0).mean(dim=0)
    return results.numpy()


def post_process_with_prob(prob: np.ndarray, threshold: float = 0.45) -> tuple[list[str], list[str]]:
    results_prob: dict[str, float] = dict(zip(_LESION_TO_CAMID.keys(), prob))
    # results_prob: dict[str, float] = {k: float(v) for k, v in sorted(results_prob.items()) if k in _TARGET_LESIONS}
    results_prob: dict[str, float] = {k: float(v) for k, v in results_prob.items() if k in _TARGET_LESIONS}

    # results_label = [f'{k}: {v:.2f}%' for k, v in results_prob.items() if v > threshold]
    # NOTE: 급하게 max값만 사용하도록 수정
    results_max_key = max(results_prob, key=results_prob.get)
    results_max_val = results_prob[results_max_key]
    if results_max_val < threshold:
        results_label = ['normal']
    else:
        results_label = [f'{results_max_key}: {results_max_val*100:.2f}']
    converted_results_prob = [f'{k}: {v*100:.2f}' for k, v in results_prob.items()]
    return converted_results_prob, results_label

def post_prcess_cam(cam: np.ndarray) -> np.ndarray:
    def min_max_normalize(cam: np.ndarray) -> np.ndarray:
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam
    cam = min_max_normalize(cam)
    # cam = np.where(cam > 0.5, cam, 0)
    cam = plt.cm.jet(cam)[:, :, :3]
    cam = (cam * 255).astype(np.uint8)
    return cam

def get_cam_per_lesion(detected_labels: list[str], normailized_images: list[torch.Tensor],
                       target_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    cam_targets = [[ClassifierOutputTarget(_LESION_TO_CAMID[lesion])] for lesion in detected_labels]
    # print(len(normailized_images))
    cam_images = {}
    for i, (lesion, target) in enumerate(zip(detected_labels, cam_targets)):
        cams = []
        for j, img in enumerate(normailized_images):
            cam = _CAM(input_tensor=img, targets=target)
            # cams.append(cam.squeeze())
            if j >= (len(normailized_images) // 2):
                # flip back
                cams.append(cam.squeeze()[:, ::-1])
            else:
                cams.append(cam.squeeze())

        cams = [cv2.resize(cam, target_shape) for cam in cams]
        mean_cam = np.mean(np.stack(cams, axis=0), axis=0)
        cam_images[lesion] = post_prcess_cam(mean_cam)

    return cam_images


@torch.no_grad()
def inference_images_for_lung_seg(image: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    output = _SEG_MODEL(image).detach().cpu().squeeze().numpy()[4:6]
    output = (output - output.min()) / (output.max() - output.min())  # min-max normalization
    output = np.where(output > threshold, 1, 0)
    output = output.astype(np.uint8)
    return output


@app.route('/process_image', methods=['POST'])
def process_image_route():
    image_url = request.json.get('image_url')
    original_file_name = os.path.basename(image_url)
    present_file_name = original_file_name.split('.')[0] + '_medit.' + original_file_name.split('.')[1]
    file_name = os.path.join(DOWNLOADS_FOLDER, original_file_name)  # 파일 저장 경로 설정

    success, file_name = download_image_from_url(image_url, file_name)
    if not success:
        return jsonify({'error': 'Failed to download the image from the given URL.'})

    image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return jsonify({'error': 'Failed to load the image.'})

    # load image
    image_path = os.path.abspath(file_name)
    image: np.ndarray = load_image(image_path)
    if _VERBOSE:
        # image shape is always (H, W, C) with C=1 for grayscale images
        print(f"Image shape: {image.shape} from path: {image_path}")

    # image resize to multiple sizes for TTA
    image_sizes = [512] if not _APPLY_TTA else [384, 512, 640]
    resized_images: list[np.ndarray] = resize_images(image, image_sizes)
    if _VERBOSE:
        print(f"Resized image shapes: {[img.shape for img in resized_images]}")

    # add histogram equalization for TTA
    if _APPLY_TTA:
        equalized_images = [cv2.equalizeHist(img)[..., None] for img in resized_images]
        resized_images.extend(equalized_images)

    ####################################
    # add some augmentation for TTA
    # do some augmentation here
    ####################################

    # add flip for TTA
    if _APPLY_TTA:
        flipped_images = [np.fliplr(img) for img in resized_images]
        resized_images.extend(flipped_images)

    # image normalization
    normailized_images: list[torch.Tensor] = [normailize_image(img) for img in resized_images]
    if _VERBOSE:
        print(f"Normalized image values: {[(img.min(), img.max()) for img in normailized_images]}")

    lung_mask = inference_images_for_lung_seg(normailized_images[0] if not _APPLY_TTA else normailized_images[1],
                                              threshold=0.5)

    # get probabilities from model and post process
    results_prob: np.ndarray = inference_images(normailized_images, _MODEL)
    results_prob, results_label = post_process_with_prob(results_prob, threshold=0.5)
    if _VERBOSE:
        print(f"Results probabilities: {results_prob}")
        print(f"Results labels: {results_label}")

    detected_labels = [label.split(':')[0] for label in results_label]
    if _VERBOSE:
        print(f"Detected labels: {detected_labels}")

    s3_url = None  # 기본값으로 None 설정

    if 'normal' not in detected_labels:
        # if lesion in results_label else 'normal' then draw cam
        cam_images: dict[str, np.ndarray] = get_cam_per_lesion(detected_labels, normailized_images,
                                                               target_shape=(image.shape[1], image.shape[0]))
        draws = [cv2.cvtColor(copy.deepcopy(image), cv2.COLOR_GRAY2BGR).astype(np.uint8) for _ in detected_labels]
        overlays = {lesion: cv2.addWeighted(draw, 0.7, cam, 0.3, 0)
                    for draw, (lesion, cam) in zip(draws, cam_images.items())}

        for lesion, overlay in overlays.items():
            result_file_name = f'{Path(file_name).stem}_{lesion}.jpg'
            cv2.imwrite(result_file_name, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            uploaded = upload_file_to_s3(result_file_name, 'medit-static-files', result_file_name)
            if uploaded:
                s3_url = (f"https://medit-static-files.s3.ap-northeast-2.amazonaws.com/{result_file_name}")

    else:
        uploaded = upload_file_to_s3(file_name, 'medit-static-files', present_file_name)
        if uploaded:
            s3_url = f"https://medit-static-files.s3.ap-northeast-2.amazonaws.com/{present_file_name}"

    return jsonify({'s3_url': s3_url, 'result': results_prob, 'label': results_label})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
