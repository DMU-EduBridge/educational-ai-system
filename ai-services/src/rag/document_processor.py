from typing import List, Dict, Any
import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Document:
    """Document class for storing text content with metadata"""
    content: str
    metadata: Dict[str, Any]


class DocumentProcessor:
    """교과서 텍스트 처리 및 청킹"""

    def __init__(self):
        pass

    def load_textbook(self, file_path: str, subject: str, unit: str) -> List[Document]:
        """
        교과서 파일을 로드하고 Document 객체로 변환

        Args:
            file_path: 교과서 파일 경로
            subject: 과목명
            unit: 단원명

        Returns:
            List[Document]: 처리된 문서 리스트
        """
        try:
            file_path_obj = Path(file_path)

            if not file_path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            if file_path_obj.suffix.lower() not in ['.txt', '.md']:
                raise ValueError(f"Unsupported file format: {file_path_obj.suffix}")

            with open(file_path_obj, 'r', encoding='utf-8') as f:
                content = f.read()

            # 텍스트 전처리
            cleaned_text = self.preprocess_text(content)

            # 텍스트 청킹
            chunks = self.chunk_text(cleaned_text)

            # 메타데이터 추가
            base_metadata = {
                'subject': subject,
                'unit': unit,
                'source_file': str(file_path_obj.name)
            }

            documents = self.add_metadata(chunks, base_metadata)

            return documents

        except Exception as e:
            raise Exception(f"Error loading textbook: {str(e)}")

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        텍스트를 의미 단위로 청킹

        Args:
            text: 입력 텍스트
            chunk_size: 청크 최대 크기
            overlap: 청크 간 겹치는 문자 수

        Returns:
            List[str]: 청크 리스트
        """
        if not text.strip():
            return []

        # 문장 단위로 분할
        sentences = self._split_into_sentences(text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # 문장이 청크 크기보다 크면 강제로 분할
            if len(sentence) > chunk_size:
                # 현재 청크가 있으면 먼저 저장
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # 긴 문장을 청크 크기로 분할
                for i in range(0, len(sentence), chunk_size):
                    chunk_part = sentence[i:i + chunk_size]
                    chunks.append(chunk_part)
                continue

            # 현재 청크에 문장을 추가했을 때 크기 계산
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence

            # 청크 크기 초과 시 새로운 청크 시작
            if len(test_chunk) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())

                # 오버랩 적용
                if overlap > 0 and len(current_chunk) > overlap:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk = test_chunk

        # 마지막 청크 추가
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def preprocess_text(self, text: str) -> str:
        """
        텍스트 전처리 (불필요한 문자 제거, 정규화)

        Args:
            text: 원본 텍스트

        Returns:
            str: 전처리된 텍스트
        """
        # 여러 공백을 하나로 변환
        text = re.sub(r'\s+', ' ', text)

        # 연속된 줄바꿈 제거
        text = re.sub(r'\n\s*\n', '\n', text)

        # 특수 문자 정리 (기본적인 정리만)
        text = re.sub(r'[^\w\s가-힣.,!?()-]', '', text)

        # 앞뒤 공백 제거
        text = text.strip()

        return text

    def add_metadata(self, chunks: List[str], metadata: Dict[str, Any]) -> List[Document]:
        """
        청크에 메타데이터 추가

        Args:
            chunks: 텍스트 청크 리스트
            metadata: 기본 메타데이터

        Returns:
            List[Document]: 메타데이터가 추가된 Document 리스트
        """
        documents = []

        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'chunk_size': len(chunk),
                'total_chunks': len(chunks)
            })

            documents.append(Document(
                content=chunk,
                metadata=chunk_metadata
            ))

        return documents

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        텍스트를 문장 단위로 분할

        Args:
            text: 입력 텍스트

        Returns:
            List[str]: 문장 리스트
        """
        # 한국어 문장 종결 표시를 기준으로 분할
        sentence_endings = r'[.!?。]'
        sentences = re.split(sentence_endings, text)

        # 빈 문장 제거 및 정리
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences