"""
[유틸리티] 이미지 처리 함수 모음
리사이징, Lab 변환, 명암 강화 등의 전처리 기능
"""

import cv2
import numpy as np
from PIL import Image


def resize_image(image, size=(640, 480)):
    """
    이미지를 목표 크기로 리사이징
    
    Args:
        image (np.ndarray): 입력 이미지
        size (tuple): 목표 크기 (가로, 세로)
        
    Returns:
        np.ndarray: 리사이징된 이미지
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def convert_to_lab(image):
    """
    BGR 이미지를 Lab 색공간으로 변환
    
    Args:
        image (np.ndarray): 입력 이미지 (BGR)
        
    Returns:
        np.ndarray: Lab 색공간 이미지
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


def normalize_image(image, mean=None, std=None):
    """
    이미지를 [0, 1] 범위로 정규화
    
    Args:
        image (np.ndarray): 입력 이미지
        mean (list): 정규화용 평균값
        std (list): 정규화용 표준편차값
        
    Returns:
        np.ndarray: 정규화된 이미지
    """
    img_float = image.astype(np.float32) / 255.0
    
    if mean is not None and std is not None:
        mean = np.array(mean)
        std = np.array(std)
        img_float = (img_float - mean) / std
    
    return img_float


def enhance_contrast(image, clip_limit=2.0, grid_size=(8, 8)):
    """
    CLAHE (명응 적응형 히스토그램 균등화)를 이용한 명암 강화
    
    Args:
        image (np.ndarray): 입력 이미지 (BGR)
        clip_limit (float): 명암 강화 한계값
        grid_size (tuple): CLAHE 타일 크기
        
    Returns:
        np.ndarray: 명암이 강화된 이미지
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_enhanced = clahe.apply(l_channel)
    
    lab[:, :, 0] = l_enhanced
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def histogram_equalization(image):
    """
    히스토그램 균등화 적용
    
    Args:
        image (np.ndarray): 입력 이미지 (BGR)
        
    Returns:
        np.ndarray: 균등화된 이미지
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    l_channel = cv2.equalizeHist(l_channel)
    
    lab[:, :, 0] = l_channel
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
