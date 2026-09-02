"""
[Upgraded] MediaPipe Face Mesh 기반 눈 검출 및 크롭
정방형 크롭으로 EfficientNet 입력에 최적화
"""

import cv2
import numpy as np
import mediapipe as mp
# [LEGACY YOLO] from ultralytics import YOLO
import config as config


class EyeDetector:
    """
    MediaPipe Face Mesh를 이용한 눈 검출기
    - 좌안(OS): LEFT_EYE indices
    - 우안(OD): RIGHT_EYE indices
    """
    
    # MediaPipe Face Mesh landmark indices
    LEFT_EYE_INDICES = [33, 133, 157, 158, 159, 160, 161, 246, 173, 153, 154, 155, 144, 145, 163, 7]
    RIGHT_EYE_INDICES = [362, 263, 384, 385, 386, 387, 388, 466, 398, 380, 381, 382, 373, 374, 390, 249]

    def _build_square_eye_bbox(self, points, image_width, image_height, margin_ratio=0.25):
        """눈 랜드마크를 감싸는 정방형 bbox를 생성하고 이미지 경계 내로 보정한다."""
        if points is None or len(points) == 0:
            return None

        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)

        eye_width = float(x_max - x_min)
        eye_height = float(y_max - y_min)
        base_side = max(eye_width, eye_height)
        if base_side <= 0:
            return None

        # 양쪽으로 20~30% 정도 여유를 주기 위해 기본 크기를 확장한다.
        side_length = int(np.ceil(base_side * (1.0 + (2.0 * float(margin_ratio)))))
        side_length = max(2, min(side_length, int(image_width), int(image_height)))

        cx = float(x_min + x_max) / 2.0
        cy = float(y_min + y_max) / 2.0
        half_side = side_length / 2.0

        x1 = int(round(cx - half_side))
        y1 = int(round(cy - half_side))
        x2 = x1 + side_length
        y2 = y1 + side_length

        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > image_width:
            shift = x2 - image_width
            x1 = max(0, x1 - shift)
            x2 = image_width
        if y2 > image_height:
            shift = y2 - image_height
            y1 = max(0, y1 - shift)
            y2 = image_height

        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2, y2)
    
    def __init__(self, model_path=None):
        """
        MediaPipe Face Mesh 초기화
        
        Args:
            model_path (str, optional): [LEGACY YOLO] 더 이상 사용하지 않음
        """
        # [LEGACY YOLO] self.model = YOLO(model_path)
        # [LEGACY YOLO] self.conf_threshold = config.YOLO_CONF_THRESHOLD
        # [LEGACY YOLO] self.iou_threshold = config.YOLO_IOU_THRESHOLD
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
    def detect(self, image, conf_threshold=None):
        """
        [LEGACY YOLO] 이전 YOLO 인터페이스 호환성 메서드
        
        Args:
            image (np.ndarray): 입력 이미지 (BGR)
            conf_threshold (float, optional): [LEGACY YOLO] 무시됨
            
        Returns:
            dict: { 'landmarks': [...], 'frame_height': h, 'frame_width': w }
        """
        # [LEGACY YOLO] threshold = self.conf_threshold if conf_threshold is None else conf_threshold
        # [LEGACY YOLO] results = self.model.predict(
        # [LEGACY YOLO]     image,
        # [LEGACY YOLO]     conf=threshold,
        # [LEGACY YOLO]     iou=self.iou_threshold,
        # [LEGACY YOLO]     imgsz=config.YOLO_INPUT_SIZE,
        # [LEGACY YOLO]     verbose=False
        # [LEGACY YOLO] )
        # [LEGACY YOLO] return results[0] if results else None
        
        h, w, _ = image.shape

        # BGR을 RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # MediaPipe Face Mesh 처리
        results = self.face_mesh.process(image_rgb)

        # 기본 처리에서 실패하면, 이미지 크기를 축소해 재시도합니다.
        # 모바일/고해상도 캡처에서 MediaPipe가 불안정하게 동작할 수 있어 축소 재시도가 도움이 됩니다.
        if results.multi_face_landmarks is None or len(results.multi_face_landmarks) == 0:
            try:
                for scale in (0.75, 0.5, 0.33):
                    sw = max(160, int(w * scale))
                    sh = max(120, int(h * scale))
                    small_rgb = cv2.resize(image_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
                    results = self.face_mesh.process(small_rgb)
                    if results and results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                        break
            except Exception:
                results = None

        if results is None or results.multi_face_landmarks is None or len(results.multi_face_landmarks) == 0:
            return None

        # 첫 번째 얼굴만 사용
        landmarks = results.multi_face_landmarks[0].landmark

        return {
            'landmarks': landmarks,
            'frame_height': h,
            'frame_width': w
        }
    
    def get_efficientnet_crop(self, image, landmarks, indices, target_size=(224, 224), margin_ratio=0.25):
        """
        MediaPipe 랜드마크로부터 정방형 크롭 생성 (EfficientNet 입력용)
        
        Args:
            image (np.ndarray): 원본 이미지 (BGR)
            landmarks (list): MediaPipe 랜드마크
            indices (list): 눈 부분의 랜드마크 인덱스
            target_size (tuple): 최종 리사이즈 크기 (기본값: 224x224)
            
        Returns:
            tuple | None: (224x224 크롭, square bbox) 또는 None (실패 시)
        """
        h, w, _ = image.shape
        
        try:
            # 지정된 인덱스의 랜드마크 포인트 추출
            points = np.array([
                (landmarks[i].x * w, landmarks[i].y * h)
                for i in indices
                if i < len(landmarks)
            ])
            
            if len(points) == 0:
                return None

            bbox = self._build_square_eye_bbox(points, w, h, margin_ratio=margin_ratio)
            if bbox is None:
                return None

            x1, y1, x2, y2 = bbox
            
            # 크롭
            cropped_eye = image[y1:y2, x1:x2]
            
            if cropped_eye is None or cropped_eye.size == 0:
                return None
            
            # EfficientNet 입력 크기로 리사이즈
            final_eye = cv2.resize(cropped_eye, target_size, interpolation=cv2.INTER_AREA)
            
            return final_eye, bbox
        except Exception as e:
            print(f"[ERROR] get_efficientnet_crop 실패: {e}")
            return None
    
    def crop_eyes(self, image, detection_result):
        """
        MediaPipe 검출 결과로부터 양안 크롭 추출
        
        Args:
            image (np.ndarray): 원본 이미지 (BGR)
            detection_result (dict): detect() 반환값
            
        Returns:
            list: [
                {
                    'image': cropped_eye_left,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': 1.0,
                    'side': 'LEFT_EYE'
                },
                {
                    'image': cropped_eye_right,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': 1.0,
                    'side': 'RIGHT_EYE'
                }
            ]
        """
        eye_crops = []
        
        if detection_result is None:
            return eye_crops
        
        landmarks = detection_result.get('landmarks')
        if landmarks is None:
            return eye_crops
        
        # 좌안(OS) 크롭
        left_eye_result = self.get_efficientnet_crop(
            image,
            landmarks,
            self.LEFT_EYE_INDICES,
            target_size=(224, 224)
        )

        if left_eye_result is not None:
            left_eye_crop, left_bbox = left_eye_result
            eye_crops.append({
                'image': left_eye_crop,
                'bbox': left_bbox,
                'confidence': 1.0,
                'side': 'LEFT_EYE'
            })
        
        # 우안(OD) 크롭
        right_eye_result = self.get_efficientnet_crop(
            image,
            landmarks,
            self.RIGHT_EYE_INDICES,
            target_size=(224, 224)
        )

        if right_eye_result is not None:
            right_eye_crop, right_bbox = right_eye_result
            eye_crops.append({
                'image': right_eye_crop,
                'bbox': right_bbox,
                'confidence': 1.0,
                'side': 'RIGHT_EYE'
            })
        
        return eye_crops

