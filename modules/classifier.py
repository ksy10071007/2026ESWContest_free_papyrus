"""
[2단계] EfficientNet을 이용한 질환 분류
검출된 눈 영역을 질환 카테고리로 분류 (정확도: 99.09%)
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
import threading
from collections.abc import Mapping
from pathlib import Path
from PIL import Image
import config as config

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class DiseaseClassifier:
    """
    EfficientNet을 이용한 질환 분류기
    """
    
    def __init__(self, model_path=config.CLASSIFIER_MODEL_PATH, num_classes=5):
        """
        분류기 초기화
        
        Args:
            model_path (str): 모델 가중치 경로
            num_classes (int): 질환 클래스 개수 (5개: 정상, 결막염, 포도막염, 백내장, 다래끼)
        """
        self.device = config.DEVICE  # config.py에서 설정된 device 사용
        self.num_classes = num_classes
        self.input_size = config.CLASSIFIER_INPUT_SIZE
        self.confidence_threshold = config.CLASSIFIER_CONFIDENCE_THRESHOLD
        
        # ========================================
        # [1] EfficientNet-B0 모델 로드
        # ========================================
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        
        # ========================================
        # [2] 학습된 가중치 로드
        # ========================================
        checkpoint = self._load_checkpoint(model_path)
        state_dict = self._extract_state_dict(checkpoint)
        self._load_state_dict_strict(state_dict, model_path)
        self.model.to(self.device)
        self.model.eval()
        parameter_device = next(self.model.parameters()).device
        if parameter_device != self.device:
            raise RuntimeError(
                f"Classifier device mismatch after initialization: "
                f"configured={self.device}, parameters={parameter_device}"
            )
        
        # ========================================
        # [3] 정규화 파라미터
        # ========================================
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)

        # ========================================
        # [4] Grad-CAM 초기화 (재사용)
        # ========================================
        self.target_layers = [self.model.features[-1]]
        self.grad_cam = GradCAM(model=self.model, target_layers=self.target_layers)
        self.grad_cam_lock = threading.Lock()

    def _load_checkpoint(self, model_path):
        checkpoint_path = Path(model_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")
        if checkpoint_path.stat().st_size <= 0:
            raise RuntimeError(f"Classifier checkpoint is empty: {checkpoint_path}")

        try:
            return torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except TypeError:
            return torch.load(checkpoint_path, map_location=self.device)
        except Exception as exc:
            raise RuntimeError(f"Unable to load classifier checkpoint {checkpoint_path}: {exc}") from exc

    @staticmethod
    def _extract_state_dict(checkpoint):
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError(
                f"Unsupported classifier checkpoint type: {type(checkpoint).__name__}; "
                "expected a state dict mapping"
            )

        state_dict = checkpoint
        for wrapper_key in ('state_dict', 'model_state_dict'):
            wrapped = checkpoint.get(wrapper_key)
            if isinstance(wrapped, Mapping):
                state_dict = wrapped
                break

        if not state_dict or not all(isinstance(key, str) for key in state_dict):
            raise RuntimeError('Classifier checkpoint state dict is empty or contains non-string keys')

        if all(key.startswith('module.') for key in state_dict):
            state_dict = {
                key[len('module.'):]: value
                for key, value in state_dict.items()
            }
        return state_dict

    def _load_state_dict_strict(self, state_dict, model_path):
        expected_keys = set(self.model.state_dict())
        actual_keys = set(state_dict)
        missing_keys = sorted(expected_keys - actual_keys)
        unexpected_keys = sorted(actual_keys - expected_keys)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"Classifier checkpoint keys do not match {model_path}: "
                f"missing={missing_keys[:10]} (total={len(missing_keys)}), "
                f"unexpected={unexpected_keys[:10]} (total={len(unexpected_keys)})"
            )
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Classifier checkpoint tensors do not match {model_path}: {exc}"
            ) from exc
    
    def preprocess(self, image):
        """
        분류 모델 입력을 위한 전처리
        
        Args:
            image (np.ndarray): 입력 이미지 (BGR)
            
        Returns:
            torch.Tensor: 전처리된 이미지 텐서
        """
        # ========================================
        # [1] 이미지 리사이징 (224x224)
        # ========================================
        if isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            img = image
        
        img = img.resize(self.input_size, Image.BILINEAR)
        
        # ========================================
        # [2] 텐서 변환 및 정규화
        # ========================================
        img = (
            torch.from_numpy(np.array(img))
            .permute(2, 0, 1)
            .float()
            .to(self.device)
            / 255.0
        )
        
        # ImageNet 표준 정규화
        img = (img - self.mean) / self.std
        return img.unsqueeze(0)

    def _prepare_cam_base_image(self, image):
        """
        Grad-CAM overlay를 위한 RGB float 이미지 준비

        Returns:
            np.ndarray: shape=(H, W, 3), range=[0,1], RGB
        """
        if isinstance(image, np.ndarray):
            resized_bgr = cv2.resize(image, self.input_size, interpolation=cv2.INTER_CUBIC)
            rgb_uint8 = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        else:
            pil_img = image.resize(self.input_size, Image.BILINEAR)
            rgb_uint8 = np.array(pil_img)

        rgb_float = rgb_uint8.astype(np.float32) / 255.0
        return np.clip(rgb_float, 0.0, 1.0)

    def _generate_cam_image(self, input_tensor, original_image, predicted_class):
        """
        예측 클래스 기준 Grad-CAM heatmap 생성 + 원본과 overlay

        Returns:
            np.ndarray | None: OpenCV BGR 이미지
        """
        try:
            targets = [ClassifierOutputTarget(int(predicted_class))]
            with self.grad_cam_lock:
                grayscale_cam = self.grad_cam(input_tensor=input_tensor, targets=targets)

            if grayscale_cam is None or len(grayscale_cam) == 0:
                return None

            cam_base_rgb = self._prepare_cam_base_image(original_image)
            cam_rgb = show_cam_on_image(cam_base_rgb, grayscale_cam[0], use_rgb=True)
            cam_bgr = cv2.cvtColor(cam_rgb, cv2.COLOR_RGB2BGR)
            return cam_bgr
        except Exception:
            return None

    def classify_with_details(self, image, generate_cam=True):
        """
        눈 이미지 분류 + (옵션) Grad-CAM 생성

        Args:
            image (np.ndarray): 눈 크롭 이미지 (BGR)
            generate_cam (bool): Grad-CAM 생성 여부

        Returns:
            dict: 분류 결과 + heatmap 이미지
        """
        input_tensor = self.preprocess(image).to(self.device)
        parameter_device = next(self.model.parameters()).device
        if input_tensor.device != parameter_device:
            raise RuntimeError(
                f"Classifier input/model device mismatch: "
                f"input={input_tensor.device}, parameters={parameter_device}"
            )

        with torch.inference_mode():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        predicted_class = probabilities.argmax().item()
        confidence = probabilities[predicted_class].item()

        heatmap_image = None
        if generate_cam:
            heatmap_image = self._generate_cam_image(input_tensor, image, predicted_class)

        if (
            config.CUDA_EMPTY_CACHE_AFTER_ANALYSIS
            and self.device.type == 'cuda'
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()

        return {
            'class': predicted_class,
            'disease': config.DISEASE_CLASSES.get(predicted_class, '미확인'),
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy().tolist(),
            'heatmap_image': heatmap_image
        }
    
    def classify(self, image, generate_cam=True):
        """
        눈 이미지 분류 (요청 포맷)
        
        Args:
            image (np.ndarray): 눈 크롭 이미지
            generate_cam (bool): Grad-CAM 생성 여부
            
        Returns:
            tuple: (disease_name, confidence, heatmap_image_array)
        """
        details = self.classify_with_details(image, generate_cam=generate_cam)
        return details['disease'], details['confidence'], details['heatmap_image']
