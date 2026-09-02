"""
[핵심 알고리즘] 눈병 진단 시스템 모듈
- detector: YOLO 눈 검출
- classifier: 질환 분류
- analyzer: 홍채 제거 및 충혈도 분석
"""

__all__ = ['EyeDetector', 'DiseaseClassifier', 'EyeAnalyzer']


def __getattr__(name):
    if name == 'EyeDetector':
        from .detector import EyeDetector

        return EyeDetector
    if name == 'DiseaseClassifier':
        from .classifier import DiseaseClassifier

        return DiseaseClassifier
    if name == 'EyeAnalyzer':
        from .analyzer import EyeAnalyzer

        return EyeAnalyzer
    raise AttributeError(f"module 'modules' has no attribute '{name}'")
