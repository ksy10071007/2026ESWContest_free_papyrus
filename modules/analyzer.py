"""
[3단계] 눈 분석 - 홍채 제거 및 충혈도 산출
눈 영역의 충혈도를 계산하여 질환 진단에 보조
"""

import cv2
import numpy as np
import config as config


class EyeAnalyzer:
    """
    눈 영역 분석을 위한 클래스
    홍채 제거 및 충혈도 산출 기능 포함
    """
    
    def __init__(self):
        """
        분석기 초기화
        """
        self.iris_threshold = config.IRIS_THRESHOLD
        self.iris_removal_enabled = config.IRIS_REMOVAL_ENABLED
    
    def remove_iris(self, image):
        """
        눈 이미지에서 홍채 영역 제거
        
        Args:
            image (np.ndarray): 눈 영역 이미지 (BGR)
            
        Returns:
            np.ndarray: 홍채가 제거된 이미지 (평균 색상으로 대체)
        """
        if not self.iris_removal_enabled:
            return image
        
        # ========================================
        # [1] HSV 색공간 변환 (홍채 검출 용이)
        # ========================================
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # ========================================
        # [2] 홍채 영역 마스크 생성 (어두운 영역)
        # ========================================
        lower_iris = np.array([0, 30, 0])
        upper_iris = np.array([180, 255, 100])
        mask = cv2.inRange(hsv, lower_iris, upper_iris)
        
        # ========================================
        # [3] 모폴로지 연산으로 마스크 정제
        # ========================================
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # ========================================
        # [4] 홍채를 주변 평균 색상으로 대체
        # ========================================
        result = image.copy()
        iris_region = result[mask > 0]
        if len(iris_region) > 0:
            mean_color = iris_region.mean(axis=0).astype(np.uint8)
            result[mask > 0] = mean_color
        
        return result
    
    def calculate_redness(self, image):
        """
        눈 영역의 충혈도 점수 계산
        
        Formula: Redness_Score = (1/n) * Σ|a_i - 128|
        where:
            - a_i: Lab 색공간의 a* 채널 값
            - 128: 중립 색상 (빨강도 및 초록도 없음)
            - n: 총 픽셀 개수
        
        Args:
            image (np.ndarray): 눈 영역 이미지 (BGR)
            
        Returns:
            float: 충혈도 점수 (0~1, 1에 가까울수록 충혈)
        """
        # ========================================
        # [1] BGR에서 Lab 색공간으로 변환
        # ========================================
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # ========================================
        # [2] a* 채널 추출 (빨강-초록 축)
        # 0=초록색, 128=중립색, 255=빨강색
        # ========================================
        a_channel = lab[:, :, 1].astype(np.float32)
        
        # ========================================
        # [3] 충혈도 계산: (1/n) * Σ|a_i - 128|
        # ========================================
        a_values = a_channel.flatten()
        n = len(a_values)
        
        if n == 0:
            return 0.0
        
        # 중립값 128으로부터의 절대 편차 합
        redness_sum = np.sum(np.abs(a_values - 128.0))
        
        # 정규화: 최대값 (128 * n)으로 나누어 0~1 범위로 스케일링
        redness_score = redness_sum / (128.0 * n)
        
        # [0, 1] 범위로 클리핑
        redness_score = np.clip(redness_score, 0.0, 1.0)
        
        return float(redness_score)
    
    def analyze(self, image):
        """
        눈 영역 전체 분석 수행
        
        Args:
            image (np.ndarray): 눈 영역 이미지 (BGR)
            
        Returns:
            dict: 분석 결과 (충혈도, 처리된 이미지 포함)
        """
        # ========================================
        # [1] 홍채 제거
        # ========================================
        processed_image = self.remove_iris(image)
        
        # ========================================
        # [2] 충혈도 계산
        # ========================================
        redness_score = self.calculate_redness(processed_image)
        
        return {
            'redness': redness_score,
            'processed_image': processed_image,
            'iris_removed': self.iris_removal_enabled
        }
