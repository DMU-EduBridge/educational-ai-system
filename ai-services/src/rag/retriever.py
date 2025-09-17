from typing import List, Dict, Optional, Any
import logging
from collections import Counter
import re

from .vector_store import VectorStore
from .embeddings import EmbeddingsManager
from .document_processor import Document


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
        self.logger = logging.getLogger(__name__)

    def retrieve_context(self,
                        query: str,
                        subject: Optional[str] = None,
                        unit: Optional[str] = None,
                        k: int = 3) -> List[str]:
        """
        쿼리에 대한 관련 컨텍스트 검색

        Args:
            query: 검색 쿼리
            subject: 과목 필터
            unit: 단원 필터
            k: 반환할 결과 수

        Returns:
            List[str]: 검색된 컨텍스트 텍스트 리스트
        """
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embeddings_manager.generate_single_embedding(query)

            # 메타데이터 필터 준비
            filter_metadata = {}
            if subject:
                filter_metadata['subject'] = subject
            if unit:
                filter_metadata['unit'] = unit

            # 벡터 검색 수행
            documents = self.vector_store.similarity_search_by_embedding(
                query_embedding=query_embedding,
                k=k * 2,  # 더 많이 검색하여 후처리에서 필터링
                filter_metadata=filter_metadata if filter_metadata else None
            )

            # 결과 랭킹 및 중복 제거
            ranked_documents = self.rank_results(documents, query)

            # 상위 k개 선택
            top_documents = ranked_documents[:k]

            # 컨텍스트 텍스트만 반환
            contexts = [doc.content for doc in top_documents]

            self.logger.info(f"Retrieved {len(contexts)} contexts for query: {query[:50]}...")
            return contexts

        except Exception as e:
            self.logger.error(f"Error retrieving context: {str(e)}")
            raise

    def rank_results(self, results: List[Document], query: str) -> List[Document]:
        """
        검색 결과 랭킹 및 중복 제거

        Args:
            results: 검색된 Document 리스트
            query: 원본 쿼리

        Returns:
            List[Document]: 랭킹된 Document 리스트
        """
        if not results:
            return []

        try:
            # 중복 제거 (내용 기반)
            unique_documents = self._remove_duplicates(results)

            # 쿼리 키워드 추출
            query_keywords = self._extract_keywords(query)

            # 각 문서에 점수 부여
            scored_documents = []
            for doc in unique_documents:
                score = self._calculate_relevance_score(doc, query_keywords)
                scored_documents.append((doc, score))

            # 점수순으로 정렬
            scored_documents.sort(key=lambda x: x[1], reverse=True)

            # Document만 반환
            ranked_documents = [doc for doc, score in scored_documents]

            self.logger.info(f"Ranked {len(ranked_documents)} documents")
            return ranked_documents

        except Exception as e:
            self.logger.error(f"Error ranking results: {str(e)}")
            return results

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

    def get_similar_content(self,
                          query: str,
                          subject: Optional[str] = None,
                          unit: Optional[str] = None,
                          similarity_threshold: float = 0.7,
                          max_results: int = 10) -> List[Dict[str, Any]]:
        """
        유사도 임계값을 적용한 컨텍스트 검색

        Args:
            query: 검색 쿼리
            subject: 과목 필터
            unit: 단원 필터
            similarity_threshold: 유사도 임계값
            max_results: 최대 결과 수

        Returns:
            List[Dict]: 유사도 정보가 포함된 결과 리스트
        """
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embeddings_manager.generate_single_embedding(query)

            # 메타데이터 필터 준비
            filter_metadata = {}
            if subject:
                filter_metadata['subject'] = subject
            if unit:
                filter_metadata['unit'] = unit

            # 벡터 검색 수행
            documents = self.vector_store.similarity_search_by_embedding(
                query_embedding=query_embedding,
                k=max_results,
                filter_metadata=filter_metadata if filter_metadata else None
            )

            # 임계값 필터링
            filtered_results = []
            for doc in documents:
                similarity_score = doc.metadata.get('similarity_score', 0)
                if similarity_score >= similarity_threshold:
                    filtered_results.append({
                        'content': doc.content,
                        'metadata': doc.metadata,
                        'similarity_score': similarity_score
                    })

            self.logger.info(f"Found {len(filtered_results)} documents above threshold {similarity_threshold}")
            return filtered_results

        except Exception as e:
            self.logger.error(f"Error in similarity search: {str(e)}")
            raise

    def _remove_duplicates(self, documents: List[Document]) -> List[Document]:
        """
        내용 기반 중복 문서 제거

        Args:
            documents: Document 리스트

        Returns:
            List[Document]: 중복이 제거된 Document 리스트
        """
        seen_contents = set()
        unique_documents = []

        for doc in documents:
            # 내용의 해시값으로 중복 체크
            content_hash = hash(doc.content.strip())

            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_documents.append(doc)

        return unique_documents

    def _extract_keywords(self, text: str) -> List[str]:
        """
        텍스트에서 키워드 추출

        Args:
            text: 입력 텍스트

        Returns:
            List[str]: 추출된 키워드 리스트
        """
        # 한국어 키워드 추출을 위한 간단한 방법
        # 실제 프로덕션에서는 형태소 분석기 사용 권장

        # 특수문자 제거 및 단어 분리
        words = re.findall(r'[가-힣a-zA-Z]+', text.lower())

        # 불용어 제거 (간단한 버전)
        stop_words = {'은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '도', '로', '으로', '에서'}
        keywords = [word for word in words if word not in stop_words and len(word) > 1]

        return keywords

    def _calculate_relevance_score(self, document: Document, query_keywords: List[str]) -> float:
        """
        문서와 쿼리 키워드 간의 관련성 점수 계산

        Args:
            document: Document 객체
            query_keywords: 쿼리 키워드 리스트

        Returns:
            float: 관련성 점수
        """
        if not query_keywords:
            return document.metadata.get('similarity_score', 0)

        # 기본 유사도 점수
        base_score = document.metadata.get('similarity_score', 0)

        # 키워드 매칭 점수
        doc_text = document.content.lower()
        keyword_matches = sum(1 for keyword in query_keywords if keyword in doc_text)
        keyword_score = keyword_matches / len(query_keywords) if query_keywords else 0

        # 메타데이터 점수 (같은 과목/단원에서 높은 점수)
        metadata_score = 0
        if document.metadata.get('subject'):
            metadata_score += 0.1
        if document.metadata.get('unit'):
            metadata_score += 0.1

        # 최종 점수 계산
        final_score = base_score * 0.7 + keyword_score * 0.2 + metadata_score * 0.1

        return final_score