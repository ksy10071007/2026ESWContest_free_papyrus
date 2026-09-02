"""
[유틸리티] 로깅 및 기록 유틸리티
진단 결과를 CSV/DB에 저장하는 기능
"""

import csv
import os
from datetime import datetime
import json
import config as config


class ResultLogger:
    """
    진단 결과 로거
    CSV 또는 데이터베이스 형식으로 결과 저장
    """
    
    def __init__(self, log_format='csv'):
        """
        로거 초기화
        
        Args:
            log_format (str): 'csv' 또는 'db' 형식
        """
        self.log_format = log_format
        self.log_dir = config.LOG_DIR
        os.makedirs(self.log_dir, exist_ok=True)
        
        if self.log_format == 'csv':
            self.csv_file = os.path.join(self.log_dir, 'diagnosis_log.csv')
            self._init_csv()
    
    def _init_csv(self):
        """
        CSV 파일 초기화 (헤더 작성)
        파일이 없으면 새로 생성
        """
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    '타임스탬프',
                    '환자ID',
                    '왼쪽_눈_질환',
                    '왼쪽_눈_신뢰도',
                    '왼쪽_눈_충혈도',
                    '오른쪽_눈_질환',
                    '오른쪽_눈_신뢰도',
                    '오른쪽_눈_충혈도',
                    '비고'
                ])
    
    def log_result(self, result_data):
        """
        진단 결과 기록
        
        Args:
            result_data (dict): 진단 정보 딕셔너리
        """
        if self.log_format == 'csv':
            self._log_to_csv(result_data)
        elif self.log_format == 'db':
            self._log_to_db(result_data)
    
    def _log_to_csv(self, result_data):
        """
        결과를 CSV 파일에 기록
        """
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                result_data.get('patient_id', 'N/A'),
                result_data.get('left_eye', {}).get('disease', 'N/A'),
                result_data.get('left_eye', {}).get('confidence', 0),
                result_data.get('left_eye', {}).get('redness', 0),
                result_data.get('right_eye', {}).get('disease', 'N/A'),
                result_data.get('right_eye', {}).get('confidence', 0),
                result_data.get('right_eye', {}).get('redness', 0),
                result_data.get('notes', '')
            ])
    
    def _log_to_db(self, result_data):
        """
        결과를 데이터베이스에 기록 (향후 구현)
        """
        # TODO: 데이터베이스 로깅 구현
        pass
    
    def get_patient_history(self, patient_id):
        """
        환자의 진단 이력 조회
        
        Args:
            patient_id (str): 환자 ID
            
        Returns:
            list: 환자의 이전 진단 결과 리스트
        """
        if self.log_format != 'csv':
            return []
        
        history = []
        if os.path.exists(self.csv_file):
            with open(self.csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('환자ID') == patient_id:
                        history.append(row)
        
        return history
