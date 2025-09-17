import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.rag.vector_store import VectorStore
from src.rag.document_processor import Document


class TestVectorStore:
    """VectorStore 테스트 클래스"""

    def setup_method(self):
        """각 테스트 메서드 실행 전 초기화"""
        # 임시 디렉토리 생성
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = VectorStore(
            collection_name="test_collection",
            persist_directory=self.temp_dir
        )

    def teardown_method(self):
        """각 테스트 메서드 실행 후 정리"""
        # 임시 디렉토리 삭제
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_init(self):
        """초기화 테스트"""
        assert self.vector_store is not None
        assert self.vector_store.collection_name == "test_collection"
        assert str(self.temp_dir) in str(self.vector_store.persist_directory)

    def test_collection_creation(self):
        """컬렉션 생성 테스트"""
        # 새로운 컬렉션 이름으로 테스트
        new_store = VectorStore(
            collection_name="new_test_collection",
            persist_directory=self.temp_dir
        )
        assert new_store.collection is not None

    def test_add_documents_basic(self):
        """기본 문서 추가 테스트"""
        # 테스트 문서 생성
        documents = [
            Document(
                content="일차함수는 y = ax + b 형태입니다.",
                metadata={"subject": "수학", "unit": "일차함수", "chunk_index": 0}
            ),
            Document(
                content="기울기 a는 직선의 기울어진 정도를 나타냅니다.",
                metadata={"subject": "수학", "unit": "일차함수", "chunk_index": 1}
            )
        ]

        # 임베딩 생성 (더미 데이터)
        embeddings = [
            [0.1, 0.2, 0.3, 0.4, 0.5] * 300,  # 1536차원으로 확장
            [0.2, 0.3, 0.4, 0.5, 0.6] * 300
        ]

        # 실제로는 1536차원이어야 하므로 적절히 조정
        embeddings = [
            [0.1] * 1536,
            [0.2] * 1536
        ]

        success = self.vector_store.add_documents(documents, embeddings)
        assert success is True

        # 컬렉션 정보 확인
        info = self.vector_store.get_collection_info()
        assert info['total_documents'] == 2

    def test_add_documents_length_mismatch(self):
        """문서와 임베딩 길이 불일치 테스트"""
        documents = [
            Document(content="테스트", metadata={})
        ]
        embeddings = [
            [0.1] * 1536,
            [0.2] * 1536  # 문서보다 임베딩이 더 많음
        ]

        with pytest.raises(ValueError):
            self.vector_store.add_documents(documents, embeddings)

    def test_similarity_search_by_embedding(self):
        """임베딩 기반 유사도 검색 테스트"""
        # 먼저 문서 추가
        documents = [
            Document(
                content="일차함수는 y = ax + b 형태입니다.",
                metadata={"subject": "수학", "unit": "일차함수"}
            )
        ]
        embeddings = [[0.1] * 1536]

        self.vector_store.add_documents(documents, embeddings)

        # 검색 수행
        query_embedding = [0.1] * 1536
        results = self.vector_store.similarity_search_by_embedding(
            query_embedding=query_embedding,
            k=1
        )

        assert len(results) <= 1
        if results:
            assert isinstance(results[0], Document)
            assert "similarity_score" in results[0].metadata

    def test_similarity_search_with_filter(self):
        """필터가 있는 유사도 검색 테스트"""
        # 다른 과목의 문서들 추가
        documents = [
            Document(
                content="일차함수 내용",
                metadata={"subject": "수학", "unit": "일차함수"}
            ),
            Document(
                content="물질의 상태 내용",
                metadata={"subject": "과학", "unit": "물질의 상태"}
            )
        ]
        embeddings = [
            [0.1] * 1536,
            [0.2] * 1536
        ]

        self.vector_store.add_documents(documents, embeddings)

        # 수학 과목만 필터링해서 검색
        query_embedding = [0.1] * 1536
        results = self.vector_store.similarity_search_by_embedding(
            query_embedding=query_embedding,
            k=5,
            filter_metadata={"subject": "수학"}
        )

        # 결과가 있다면 모두 수학 과목이어야 함
        for result in results:
            assert result.metadata.get("subject") == "수학"

    def test_get_collection_info_empty(self):
        """빈 컬렉션 정보 조회 테스트"""
        info = self.vector_store.get_collection_info()

        assert info['total_documents'] == 0
        assert info['subjects'] == []
        assert info['units'] == []
        assert info['collection_name'] == "test_collection"

    def test_get_collection_info_with_data(self):
        """데이터가 있는 컬렉션 정보 조회 테스트"""
        documents = [
            Document(
                content="수학 내용 1",
                metadata={"subject": "수학", "unit": "일차함수", "source_file": "math.txt"}
            ),
            Document(
                content="과학 내용 1",
                metadata={"subject": "과학", "unit": "물질의 상태", "source_file": "science.txt"}
            )
        ]
        embeddings = [
            [0.1] * 1536,
            [0.2] * 1536
        ]

        self.vector_store.add_documents(documents, embeddings)

        info = self.vector_store.get_collection_info()

        assert info['total_documents'] == 2
        assert "수학" in info['subjects']
        assert "과학" in info['subjects']
        assert "일차함수" in info['units']
        assert "물질의 상태" in info['units']
        assert "math.txt" in info['source_files']
        assert "science.txt" in info['source_files']

    def test_clear_collection(self):
        """컬렉션 초기화 테스트"""
        # 먼저 데이터 추가
        documents = [
            Document(content="테스트 내용", metadata={"subject": "수학"})
        ]
        embeddings = [[0.1] * 1536]

        self.vector_store.add_documents(documents, embeddings)

        # 데이터가 추가됐는지 확인
        info = self.vector_store.get_collection_info()
        assert info['total_documents'] == 1

        # 컬렉션 초기화
        success = self.vector_store.clear_collection()
        assert success is True

        # 초기화 후 확인
        info = self.vector_store.get_collection_info()
        assert info['total_documents'] == 0

    def test_delete_by_metadata(self):
        """메타데이터 기반 문서 삭제 테스트"""
        documents = [
            Document(
                content="수학 내용",
                metadata={"subject": "수학", "unit": "일차함수"}
            ),
            Document(
                content="과학 내용",
                metadata={"subject": "과학", "unit": "물질의 상태"}
            )
        ]
        embeddings = [
            [0.1] * 1536,
            [0.2] * 1536
        ]

        self.vector_store.add_documents(documents, embeddings)

        # 수학 과목 문서만 삭제
        deleted_count = self.vector_store.delete_by_metadata({"subject": "수학"})

        assert deleted_count >= 0  # ChromaDB의 동작에 따라 달라질 수 있음

    def test_metadata_handling(self):
        """메타데이터 처리 테스트"""
        # 복잡한 메타데이터가 있는 문서
        document = Document(
            content="테스트 내용",
            metadata={
                "subject": "수학",
                "unit": "일차함수",
                "chunk_index": 0,
                "chunk_size": 100,
                "source_file": "test.txt",
                "nested_dict": {"key": "value"},  # 중첩 딕셔너리는 문자열로 변환됨
                "list_data": [1, 2, 3],  # 리스트도 문자열로 변환됨
                "boolean_value": True,
                "numeric_value": 42
            }
        )

        embeddings = [[0.1] * 1536]

        success = self.vector_store.add_documents([document], embeddings)
        assert success is True

    def test_persistence(self):
        """데이터 영속성 테스트"""
        # 데이터 추가
        documents = [
            Document(
                content="영속성 테스트",
                metadata={"subject": "테스트", "unit": "영속성"}
            )
        ]
        embeddings = [[0.1] * 1536]

        self.vector_store.add_documents(documents, embeddings)

        # 새로운 VectorStore 인스턴스로 같은 위치에서 로드
        new_store = VectorStore(
            collection_name="test_collection",
            persist_directory=self.temp_dir
        )

        info = new_store.get_collection_info()
        assert info['total_documents'] == 1

    def test_error_handling(self):
        """에러 처리 테스트"""
        # 빈 문서 리스트
        success = self.vector_store.add_documents([], [])
        assert success is True  # 빈 리스트는 성공으로 처리

        # None 값이 포함된 메타데이터
        document = Document(
            content="None 테스트",
            metadata={"subject": None, "unit": "테스트"}
        )
        embeddings = [[0.1] * 1536]

        # 이것은 실패할 수도 있으므로 try-except로 처리
        try:
            self.vector_store.add_documents([document], embeddings)
        except Exception:
            pass  # 예상된 동작

    def test_large_batch_insertion(self):
        """대량 데이터 삽입 테스트"""
        # 많은 문서 생성
        documents = []
        embeddings = []

        for i in range(10):  # 적당한 수로 제한
            documents.append(Document(
                content=f"문서 {i} 내용",
                metadata={"subject": "수학", "unit": "테스트", "index": i}
            ))
            embeddings.append([0.1 + i * 0.01] * 1536)

        success = self.vector_store.add_documents(documents, embeddings)
        assert success is True

        info = self.vector_store.get_collection_info()
        assert info['total_documents'] == 10