import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import os

from src.rag.document_processor import DocumentProcessor
from src.rag.embeddings import EmbeddingsManager
from src.rag.vector_store import VectorStore
from src.rag.retriever import RAGRetriever
from src.models.llm_client import LLMClient
from src.models.question_generator import QuestionGenerator
from src.main import RAGPipeline
from src.utils.config import get_test_settings


class TestIntegration:
    """통합 테스트 클래스"""

    def setup_method(self):
        """각 테스트 메서드 실행 전 초기화"""
        # 임시 디렉토리 생성
        self.temp_dir = tempfile.mkdtemp()

        # 테스트 설정 로드
        self.settings = get_test_settings()
        self.settings.chroma_db_path = str(Path(self.temp_dir) / "vector_db")
        self.settings.cache_dir = str(Path(self.temp_dir) / "cache")

        # 테스트용 API 키 설정 (실제로는 환경변수에서 가져와야 함)
        self.settings.openai_api_key = os.getenv('OPENAI_API_KEY', 'sk-test-key-for-testing')

    def teardown_method(self):
        """각 테스트 메서드 실행 후 정리"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_document_processing_pipeline(self):
        """문서 처리 파이프라인 통합 테스트"""
        # 테스트 파일 생성
        test_content = """일차함수의 정의

일차함수는 y = ax + b (a ≠ 0) 형태로 나타낼 수 있는 함수입니다.
여기서 a는 기울기, b는 y절편을 나타냅니다.

일차함수의 그래프는 직선입니다.
기울기가 양수이면 우상향하고, 음수이면 우하향합니다."""

        test_file = Path(self.temp_dir) / "test_math.txt"
        test_file.write_text(test_content, encoding='utf-8')

        # 컴포넌트 초기화
        processor = DocumentProcessor()
        vector_store = VectorStore(
            collection_name="test_integration",
            persist_directory=self.settings.chroma_db_path
        )

        # 문서 처리
        documents = processor.load_textbook(str(test_file), "수학", "일차함수")

        # 검증
        assert len(documents) > 0
        assert all(doc.metadata['subject'] == "수학" for doc in documents)
        assert all(doc.metadata['unit'] == "일차함수" for doc in documents)

        # 임베딩 및 저장 (Mock 사용)
        with patch('src.rag.embeddings.EmbeddingsManager') as mock_embeddings:
            mock_embeddings.return_value.generate_embeddings.return_value = [
                [0.1] * 1536 for _ in documents
            ]

            embeddings_manager = mock_embeddings.return_value
            embeddings = embeddings_manager.generate_embeddings([doc.content for doc in documents])

            success = vector_store.add_documents(documents, embeddings)
            assert success is True

            # 저장 확인
            info = vector_store.get_collection_info()
            assert info['total_documents'] == len(documents)

    @patch('src.rag.embeddings.EmbeddingsManager')
    @patch('src.models.llm_client.LLMClient')
    def test_question_generation_pipeline(self, mock_llm, mock_embeddings):
        """문제 생성 파이프라인 통합 테스트"""
        # Mock 설정
        mock_embeddings.return_value.generate_embeddings.return_value = [[0.1] * 1536]
        mock_embeddings.return_value.generate_single_embedding.return_value = [0.1] * 1536

        mock_llm.return_value.generate_structured_response.return_value = {
            "question": "일차함수 y = 2x + 3에서 기울기는 무엇인가?",
            "options": ["1", "2", "3", "4", "5"],
            "correct_answer": 2,
            "explanation": "일차함수 y = ax + b에서 a가 기울기이므로 답은 2입니다.",
            "difficulty": "medium",
            "subject": "수학",
            "unit": "일차함수"
        }

        # 컴포넌트 초기화
        vector_store = VectorStore(
            collection_name="test_integration",
            persist_directory=self.settings.chroma_db_path
        )

        # 테스트 문서 추가
        from src.rag.document_processor import Document
        test_documents = [
            Document(
                content="일차함수는 y = ax + b 형태입니다.",
                metadata={"subject": "수학", "unit": "일차함수"}
            )
        ]

        vector_store.add_documents(test_documents, [[0.1] * 1536])

        # RAG 컴포넌트 초기화
        retriever = RAGRetriever(vector_store, mock_embeddings.return_value)
        question_generator = QuestionGenerator(mock_llm.return_value, retriever)

        # 문제 생성
        question = question_generator.generate_question("수학", "일차함수", "medium")

        # 검증
        assert question is not None
        assert question["subject"] == "수학"
        assert question["unit"] == "일차함수"
        assert question["difficulty"] == "medium"
        assert len(question["options"]) == 5
        assert 1 <= question["correct_answer"] <= 5

    @patch('src.rag.embeddings.openai.OpenAI')
    @patch('src.models.llm_client.openai.OpenAI')
    def test_rag_pipeline_end_to_end(self, mock_llm_openai, mock_embed_openai):
        """전체 RAG 파이프라인 End-to-End 테스트"""
        # OpenAI API Mock 설정
        mock_embed_client = MagicMock()
        mock_embed_openai.return_value = mock_embed_client

        mock_embed_response = MagicMock()
        mock_embed_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_embed_client.embeddings.create.return_value = mock_embed_response

        mock_llm_client = MagicMock()
        mock_llm_openai.return_value = mock_llm_client

        mock_llm_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = """{
    "question": "일차함수 y = 2x + 3에서 기울기는?",
    "options": ["1", "2", "3", "4", "5"],
    "correct_answer": 2,
    "explanation": "기울기는 2입니다.",
    "difficulty": "medium",
    "subject": "수학",
    "unit": "일차함수"
}"""
        mock_choice.message = mock_message
        mock_llm_response.choices = [mock_choice]

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150
        mock_llm_response.usage = mock_usage

        mock_llm_client.chat.completions.create.return_value = mock_llm_response

        # 테스트 파일 생성
        test_content = "일차함수는 y = ax + b 형태입니다. 기울기 a는 직선의 기울어진 정도를 나타냅니다."
        test_file = Path(self.temp_dir) / "test_textbook.txt"
        test_file.write_text(test_content, encoding='utf-8')

        # RAGPipeline 테스트
        try:
            pipeline = RAGPipeline(self.settings)

            # 1. 교과서 처리
            process_result = pipeline.process_textbook(
                str(test_file), "수학", "일차함수"
            )

            assert process_result['status'] == 'success'
            assert process_result['processed_chunks'] > 0

            # 2. 문제 생성
            questions = pipeline.generate_questions(
                "수학", "일차함수", "medium", 1
            )

            assert len(questions) == 1
            assert questions[0]['subject'] == "수학"
            assert questions[0]['unit'] == "일차함수"

            # 3. 시스템 상태 확인
            status = pipeline.get_status()
            assert 'vector_store' in status
            assert 'llm_usage' in status
            assert 'question_generation' in status

        except Exception as e:
            # API 키가 없는 경우 등 예상된 실패
            if "Invalid or missing OpenAI API key" in str(e):
                pytest.skip("OpenAI API key not available for testing")
            else:
                raise

    def test_error_handling_integration(self):
        """에러 처리 통합 테스트"""
        # 잘못된 설정으로 RAGPipeline 초기화 시도
        bad_settings = get_test_settings()
        bad_settings.openai_api_key = "invalid-key"

        with pytest.raises(Exception):
            RAGPipeline(bad_settings)

    def test_data_persistence_integration(self):
        """데이터 영속성 통합 테스트"""
        # 첫 번째 인스턴스로 데이터 저장
        vector_store1 = VectorStore(
            collection_name="persistence_test",
            persist_directory=self.settings.chroma_db_path
        )

        from src.rag.document_processor import Document
        test_doc = Document(
            content="영속성 테스트 문서",
            metadata={"subject": "테스트", "unit": "영속성"}
        )

        vector_store1.add_documents([test_doc], [[0.1] * 1536])

        # 정보 확인
        info1 = vector_store1.get_collection_info()
        assert info1['total_documents'] == 1

        # 두 번째 인스턴스로 같은 위치에서 로드
        vector_store2 = VectorStore(
            collection_name="persistence_test",
            persist_directory=self.settings.chroma_db_path
        )

        info2 = vector_store2.get_collection_info()
        assert info2['total_documents'] == 1
        assert info2['subjects'] == info1['subjects']

    @patch('src.rag.embeddings.openai.OpenAI')
    def test_retrieval_accuracy(self, mock_openai):
        """검색 정확도 통합 테스트"""
        # Mock 설정
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # 서로 다른 임베딩 반환
        def side_effect(*args, **kwargs):
            input_texts = kwargs.get('input', [])
            mock_response = MagicMock()
            mock_response.data = []

            for i, text in enumerate(input_texts):
                mock_data = MagicMock()
                if "일차함수" in text:
                    mock_data.embedding = [0.9] * 1536  # 수학 관련 높은 유사도
                elif "물질" in text:
                    mock_data.embedding = [0.1] * 1536  # 과학 관련 낮은 유사도
                else:
                    mock_data.embedding = [0.5] * 1536  # 중간 유사도

                mock_response.data.append(mock_data)

            return mock_response

        mock_client.embeddings.create.side_effect = side_effect

        # 테스트 문서들 생성
        from src.rag.document_processor import Document

        documents = [
            Document(
                content="일차함수는 y = ax + b 형태의 함수입니다.",
                metadata={"subject": "수학", "unit": "일차함수"}
            ),
            Document(
                content="물질은 고체, 액체, 기체의 상태로 존재합니다.",
                metadata={"subject": "과학", "unit": "물질의 상태"}
            ),
            Document(
                content="일차함수의 그래프는 직선입니다.",
                metadata={"subject": "수학", "unit": "일차함수"}
            )
        ]

        # 컴포넌트 초기화
        embeddings_manager = EmbeddingsManager(api_key="test-key")
        vector_store = VectorStore(
            collection_name="retrieval_test",
            persist_directory=self.settings.chroma_db_path
        )

        # 임베딩 생성 및 저장
        embeddings = embeddings_manager.generate_embeddings([doc.content for doc in documents])
        vector_store.add_documents(documents, embeddings)

        # 검색기 초기화
        retriever = RAGRetriever(vector_store, embeddings_manager)

        # 수학 관련 쿼리로 검색
        contexts = retriever.retrieve_context(
            query="일차함수의 기울기",
            subject="수학",
            k=2
        )

        # 수학 관련 내용이 우선적으로 반환되어야 함
        assert len(contexts) > 0
        math_related = any("일차함수" in context for context in contexts)
        assert math_related

    def test_performance_integration(self):
        """성능 통합 테스트"""
        import time

        # 많은 문서로 성능 테스트
        from src.rag.document_processor import Document

        documents = []
        for i in range(50):  # 적당한 수로 제한
            documents.append(Document(
                content=f"테스트 문서 {i}의 내용입니다. 이는 성능 테스트를 위한 문서입니다.",
                metadata={"subject": "테스트", "unit": f"단원{i%5}", "index": i}
            ))

        # 시간 측정
        start_time = time.time()

        vector_store = VectorStore(
            collection_name="performance_test",
            persist_directory=self.settings.chroma_db_path
        )

        # 더미 임베딩으로 저장
        embeddings = [[0.1 + i * 0.01] * 1536 for i in range(len(documents))]
        vector_store.add_documents(documents, embeddings)

        end_time = time.time()
        processing_time = end_time - start_time

        # 성능 검증 (50개 문서 처리가 10초 이내)
        assert processing_time < 10.0

        # 검색 성능 테스트
        start_time = time.time()

        query_embedding = [0.1] * 1536
        results = vector_store.similarity_search_by_embedding(
            query_embedding=query_embedding,
            k=5
        )

        end_time = time.time()
        search_time = end_time - start_time

        # 검색이 1초 이내
        assert search_time < 1.0
        assert len(results) <= 5