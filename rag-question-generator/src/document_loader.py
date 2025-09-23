"""
Document Loader - 문서 로딩 및 전처리
다양한 형태의 텍스트 문서를 로드하고 전처리합니다.
"""

import os
import re
from pathlib import Path
from typing import Optional


class DocumentLoader:
    """문서 로딩 및 전처리"""

    @staticmethod
    def load_text_file(file_path: str) -> str:
        """
        텍스트 파일 로드

        Args:
            file_path: 파일 경로

        Returns:
            str: 로드된 텍스트

        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
            UnicodeDecodeError: 인코딩 오류가 발생한 경우
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"디렉토리입니다. 파일이 아닙니다: {file_path}")

        # 파일 크기 확인 (10MB 제한)
        file_size = file_path.stat().st_size
        if file_size > 10 * 1024 * 1024:  # 10MB
            raise ValueError(f"파일이 너무 큽니다 (10MB 초과): {file_size / 1024 / 1024:.1f}MB")

        # 다양한 인코딩으로 시도
        encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()

                # 내용이 비어있는지 확인
                if not content.strip():
                    raise ValueError("파일이 비어있습니다.")

                print(f"✅ 파일 로드 성공: {file_path} ({encoding} 인코딩)")
                return content

            except UnicodeDecodeError:
                continue

        raise UnicodeDecodeError(f"지원되는 인코딩으로 파일을 읽을 수 없습니다: {file_path}")

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        텍스트 전처리 (정규화, 정제)

        Args:
            text: 원본 텍스트

        Returns:
            str: 전처리된 텍스트
        """
        if not text:
            return ""

        # 1. 기본 정제
        # BOM 제거
        text = text.replace('\ufeff', '')

        # 탭을 공백으로 변환
        text = text.replace('\t', ' ')

        # 윈도우 스타일 줄바꿈을 유닉스 스타일로 변환
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')

        # 2. 공백 정규화
        # 여러 개의 공백을 하나로 통합
        text = re.sub(r'[ ]+', ' ', text)

        # 여러 개의 줄바꿈을 최대 2개로 제한
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 3. 특수 문자 정제
        # 보이지 않는 문자 제거
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)

        # 4. 문장 구조 개선
        # 줄 끝의 공백 제거
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)

        # 5. 최종 정제
        text = text.strip()

        return text

    @staticmethod
    def validate_document(text: str) -> bool:
        """
        문서 유효성 검증

        Args:
            text: 검증할 텍스트

        Returns:
            bool: 유효성 검사 결과
        """
        if not text or not isinstance(text, str):
            return False

        # 최소 길이 확인 (100자 이상)
        if len(text.strip()) < 100:
            print("❌ 텍스트가 너무 짧습니다 (최소 100자 필요)")
            return False

        # 최대 길이 확인 (1MB 이하)
        if len(text.encode('utf-8')) > 1024 * 1024:  # 1MB
            print("❌ 텍스트가 너무 깁니다 (최대 1MB)")
            return False

        # 의미있는 내용이 있는지 확인
        # 알파벳, 숫자, 한글이 포함되어 있는지 확인
        has_meaningful_content = bool(re.search(r'[a-zA-Z0-9가-힣]', text))
        if not has_meaningful_content:
            print("❌ 의미있는 내용이 없습니다")
            return False

        # 텍스트 다양성 확인 (같은 문자가 90% 이상 반복되지 않음)
        if len(set(text)) / len(text) < 0.1:
            print("❌ 텍스트 다양성이 부족합니다")
            return False

        return True

    @staticmethod
    def get_file_info(file_path: str) -> dict:
        """
        파일 정보 반환

        Args:
            file_path: 파일 경로

        Returns:
            dict: 파일 정보
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return {"error": "파일이 존재하지 않습니다"}

        stat = file_path.stat()

        return {
            "name": file_path.name,
            "size_bytes": stat.st_size,
            "size_kb": stat.st_size / 1024,
            "size_mb": stat.st_size / (1024 * 1024),
            "extension": file_path.suffix,
            "is_text_file": file_path.suffix.lower() in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json'],
            "modified_time": stat.st_mtime
        }

    @staticmethod
    def load_and_process(file_path: str) -> str:
        """
        파일을 로드하고 전처리까지 한번에 수행

        Args:
            file_path: 파일 경로

        Returns:
            str: 로드되고 전처리된 텍스트

        Raises:
            ValueError: 파일 검증 실패 시
        """
        print(f"📄 파일 로딩 시작: {file_path}")

        # 1. 파일 정보 확인
        file_info = DocumentLoader.get_file_info(file_path)
        if "error" in file_info:
            raise ValueError(file_info["error"])

        print(f"📊 파일 정보: {file_info['name']} ({file_info['size_kb']:.1f}KB)")

        # 2. 파일 로드
        raw_text = DocumentLoader.load_text_file(file_path)
        print(f"📝 원본 텍스트 길이: {len(raw_text):,}자")

        # 3. 전처리
        processed_text = DocumentLoader.preprocess_text(raw_text)
        print(f"🔧 전처리 후 텍스트 길이: {len(processed_text):,}자")

        # 4. 검증
        if not DocumentLoader.validate_document(processed_text):
            raise ValueError("문서 검증에 실패했습니다")

        print("✅ 문서 로딩 및 전처리 완료")
        return processed_text