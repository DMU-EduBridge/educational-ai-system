import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.rag.document_processor import DocumentProcessor, Document


class TestDocumentProcessor:
    """DocumentProcessor 테스트 클래스"""

    def setup_method(self):
        """각 테스트 메서드 실행 전 초기화"""
        self.processor = DocumentProcessor()

    def test_init(self):
        """초기화 테스트"""
        assert self.processor is not None

    def test_load_textbook_txt_file(self):
        """텍스트 파일 로딩 테스트"""
        # 임시 텍스트 파일 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("일차함수는 y = ax + b 형태입니다. 여기서 a는 기울기입니다.")
            temp_file = f.name

        try:
            documents = self.processor.load_textbook(temp_file, "수학", "일차함수")

            assert len(documents) > 0
            assert isinstance(documents[0], Document)
            assert documents[0].metadata['subject'] == "수학"
            assert documents[0].metadata['unit'] == "일차함수"
            assert "일차함수" in documents[0].content

        finally:
            Path(temp_file).unlink()

    def test_load_textbook_md_file(self):
        """마크다운 파일 로딩 테스트"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write("# 물질의 상태\n\n물질은 고체, 액체, 기체로 존재합니다.")
            temp_file = f.name

        try:
            documents = self.processor.load_textbook(temp_file, "과학", "물질의 상태")

            assert len(documents) > 0
            assert documents[0].metadata['subject'] == "과학"
            assert documents[0].metadata['unit'] == "물질의 상태"

        finally:
            Path(temp_file).unlink()

    def test_load_textbook_file_not_found(self):
        """파일이 존재하지 않을 때 예외 처리 테스트"""
        with pytest.raises(Exception) as exc_info:
            self.processor.load_textbook("nonexistent.txt", "수학", "일차함수")

        assert "File not found" in str(exc_info.value)

    def test_load_textbook_unsupported_format(self):
        """지원하지 않는 파일 형식 테스트"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
            temp_file = f.name

        try:
            with pytest.raises(Exception) as exc_info:
                self.processor.load_textbook(temp_file, "수학", "일차함수")

            assert "Unsupported file format" in str(exc_info.value)

        finally:
            Path(temp_file).unlink()

    def test_chunk_text_basic(self):
        """기본 텍스트 청킹 테스트"""
        text = "문장1입니다. 문장2입니다. 문장3입니다. 문장4입니다."
        chunks = self.processor.chunk_text(text, chunk_size=20, overlap=5)

        assert len(chunks) > 0
        assert all(len(chunk) <= 20 for chunk in chunks if chunk.strip())

    def test_chunk_text_empty(self):
        """빈 텍스트 청킹 테스트"""
        chunks = self.processor.chunk_text("", chunk_size=100)
        assert chunks == []

        chunks = self.processor.chunk_text("   ", chunk_size=100)
        assert chunks == []

    def test_chunk_text_with_overlap(self):
        """오버랩이 있는 청킹 테스트"""
        text = "가나다라마바사. 아자차카타파하. 하나둘셋넷다섯."
        chunks = self.processor.chunk_text(text, chunk_size=15, overlap=5)

        assert len(chunks) >= 2
        # 오버랩 확인을 위해 연속된 청크 간에 공통 부분이 있는지 확인

    def test_preprocess_text(self):
        """텍스트 전처리 테스트"""
        text = "  여러  공백이   있는\n\n\n텍스트입니다.  "
        processed = self.processor.preprocess_text(text)

        assert "여러 공백이 있는 텍스트입니다." in processed
        assert processed.strip() == processed  # 앞뒤 공백 제거 확인

    def test_preprocess_text_special_characters(self):
        """특수문자 처리 테스트"""
        text = "텍스트@#$%^&*()[]{}|\\=+~`"
        processed = self.processor.preprocess_text(text)

        # 기본 문자와 한글, 구두점은 유지되어야 함
        assert "텍스트" in processed

    def test_add_metadata(self):
        """메타데이터 추가 테스트"""
        chunks = ["첫 번째 청크", "두 번째 청크", "세 번째 청크"]
        metadata = {'subject': '수학', 'unit': '일차함수', 'source_file': 'test.txt'}

        documents = self.processor.add_metadata(chunks, metadata)

        assert len(documents) == 3
        for i, doc in enumerate(documents):
            assert isinstance(doc, Document)
            assert doc.metadata['subject'] == '수학'
            assert doc.metadata['unit'] == '일차함수'
            assert doc.metadata['chunk_index'] == i
            assert doc.metadata['total_chunks'] == 3
            assert doc.metadata['chunk_size'] == len(chunks[i])

    def test_add_metadata_empty_chunks(self):
        """빈 청크 리스트에 메타데이터 추가 테스트"""
        documents = self.processor.add_metadata([], {})
        assert documents == []

    def test_split_into_sentences(self):
        """문장 분할 테스트"""
        text = "첫 번째 문장입니다. 두 번째 문장입니다! 세 번째 문장입니다?"
        sentences = self.processor._split_into_sentences(text)

        assert len(sentences) == 3
        assert "첫 번째 문장입니다" in sentences[0]
        assert "두 번째 문장입니다" in sentences[1]
        assert "세 번째 문장입니다" in sentences[2]

    def test_split_into_sentences_korean_punctuation(self):
        """한국어 문장 부호 분할 테스트"""
        text = "첫 번째 문장입니다。 두 번째 문장입니다。"
        sentences = self.processor._split_into_sentences(text)

        assert len(sentences) >= 2

    def test_split_into_sentences_empty(self):
        """빈 텍스트 문장 분할 테스트"""
        sentences = self.processor._split_into_sentences("")
        assert sentences == []

        sentences = self.processor._split_into_sentences("   ")
        assert sentences == []

    def test_integration_full_process(self):
        """전체 프로세스 통합 테스트"""
        # 실제 사용 시나리오와 유사한 테스트
        content = """일차함수의 정의

일차함수는 y = ax + b (a ≠ 0) 형태로 나타낼 수 있는 함수입니다.
여기서 a는 기울기, b는 y절편을 나타냅니다.

일차함수의 그래프는 직선입니다.
기울기 a가 양수이면 우상향하고, 음수이면 우하향합니다."""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_file = f.name

        try:
            documents = self.processor.load_textbook(temp_file, "수학", "일차함수")

            # 기본 검증
            assert len(documents) > 0
            assert all(isinstance(doc, Document) for doc in documents)
            assert all(doc.metadata['subject'] == "수학" for doc in documents)
            assert all(doc.metadata['unit'] == "일차함수" for doc in documents)

            # 내용 검증
            all_content = " ".join(doc.content for doc in documents)
            assert "일차함수" in all_content
            assert "기울기" in all_content

        finally:
            Path(temp_file).unlink()

    def test_large_text_chunking(self):
        """큰 텍스트 청킹 성능 테스트"""
        # 긴 텍스트 생성
        long_text = "긴 문장입니다. " * 1000  # 약 15,000자

        chunks = self.processor.chunk_text(long_text, chunk_size=500, overlap=50)

        assert len(chunks) > 1
        assert all(len(chunk) <= 500 for chunk in chunks)

    def test_unicode_handling(self):
        """유니코드 문자 처리 테스트"""
        text = "한글과 English와 日本語와 中文이 섞인 텍스트입니다."

        processed = self.processor.preprocess_text(text)
        assert "한글과" in processed
        assert "English" in processed

        chunks = self.processor.chunk_text(text)
        assert len(chunks) > 0
        assert any("한글과" in chunk for chunk in chunks)