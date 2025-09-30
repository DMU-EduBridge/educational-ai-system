from typing import List, Dict, Optional, Any
import logging

from .vector_store import VectorStore
from .embeddings import EmbeddingsManager
from .document_processor import Document
from .re_ranker import ReRanker


class RAGRetriever:
    """컨텍스트 검색 및 랭킹"""

    def __init__(self,
                 vector_store: VectorStore,
                 embeddings_manager: EmbeddingsManager):
        """
        RAGRetriever 초기화

        Args:
            vector_store: VectorStore 인스턴스
            embeddings_manager: EmbeddingsManager 인스턴스
        """
        self.vector_store = vector_store
        self.embeddings_manager = embeddings_manager
        self.re_ranker = ReRanker()  # ReRanker 초기화
        self.logger = logging.getLogger(__name__)

    def retrieve_documents(self,
                         query: str,
                         subject: Optional[str] = None,
                         unit: Optional[str] = None,
                         k: int = 3,
                         candidates: int = 10) -> List[Document]:
        """
        쿼리에 대한 관련 문서를 검색하고 재순위화합니다.

        Args:
            query: 검색 쿼리
            subject: 과목 필터
            unit: 단원 필터
            k: 반환할 최종 문서 수
            candidates: 1차 검색할 후보 문서 수

        Returns:
            List[Document]: 재순위화된 상위 k개의 문서 리스트
        """
        try:
            # 1. 쿼리 임베딩 생성
            query_embedding = self.embeddings_manager.generate_single_embedding(query)

            # 2. 메타데이터 필터 준비
            filter_metadata = {}
            if subject:
                filter_metadata['subject'] = subject
            if unit:
                filter_metadata['unit'] = unit

            # 3. 벡터 저장소에서 후보 문서 검색
            candidate_docs = self.vector_store.similarity_search_by_embedding(
                query_embedding=query_embedding,
                k=candidates,
                filter_metadata=filter_metadata if filter_metadata else None
            )

            if not candidate_docs:
                self.logger.warning("No documents found from vector store.")
                return []

            # 4. ReRanker를 사용한 재순위화
            reranked_docs = self.re_ranker.rerank(query, candidate_docs)
            self.logger.info(f"Re-ranked {len(candidate_docs)} documents.")

            # 5. 상위 k개 문서 선택 및 반환
            top_k_docs = reranked_docs[:k]
            self.logger.info(f"Retrieved {len(top_k_docs)} documents for query: {query[:50]}...")
            
            return top_k_docs

        except Exception as e:
            self.logger.error(f"Error retrieving documents: {str(e)}")
            raise

    def format_context(self, documents: List[Document]) -> str:
        """
        문서들을 LLM 입력용 컨텍스트로 포맷팅

        Args:
            documents: Document 리스트

        Returns:
            str: 포맷팅된 컨텍스트 텍스트
        """
        if not documents:
            return ""

        try:
            formatted_parts = []

            for i, doc in enumerate(documents, 1):
                # 메타데이터 정보 추가
                metadata_info = []
                if 'subject' in doc.metadata:
                    metadata_info.append(f"과목: {doc.metadata['subject']}")
                if 'unit' in doc.metadata:
                    metadata_info.append(f"단원: {doc.metadata['unit']}")

                metadata_str = f" ({', '.join(metadata_info)})" if metadata_info else ""

                # 문서 포맷팅
                formatted_part = f"[참고자료 {i}]{metadata_str}\n{doc.content}"
                formatted_parts.append(formatted_part)

            context = "\n\n".join(formatted_parts)

            self.logger.info(f"Formatted context with {len(documents)} documents")
            return context

        except Exception as e:
            self.logger.error(f"Error formatting context: {str(e)}")
            # 오류 시 간단한 포맷으로 대체
            return "\n\n".join([doc.content for doc in documents])