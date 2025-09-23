"""
RAG Processor - RAG 처리 핵심 모듈
텍스트 문서를 벡터화하고 검색 가능하도록 처리
"""

import os
import re
from typing import List, Optional
import tiktoken
import chromadb
from chromadb.utils import embedding_functions


class RAGProcessor:
    """RAG 처리 핵심 클래스"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Args:
            chunk_size: 텍스트 청킹 크기
            overlap: 청크 간 겹치는 부분
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

        # OpenAI API 키 확인
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

        # 텍스트 분할 설정
        self.separators = ["\n\n", "\n", ".", "?", "!", " ", ""]

        # ChromaDB 클라이언트 초기화
        self.chroma_client = chromadb.PersistentClient(
            path="./ai-services/rag-question-generator/data/vector_db"
        )

        # OpenAI 임베딩 함수 설정
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.api_key,
            model_name="text-embedding-ada-002"
        )

        # 컬렉션 이름
        self.collection_name = "document_chunks"
        self.collection = None
        self.document_summary = ""

        # 토큰 계산용 인코더
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def process_document(self, text: str) -> bool:
        """
        문서를 RAG 처리하여 벡터 DB에 저장

        Args:
            text: 입력 텍스트

        Returns:
            bool: 처리 성공 여부
        """
        try:
            # 문서 전처리
            text = self._preprocess_text(text)

            # 문서 요약 저장
            self.document_summary = self._create_summary(text)

            # 텍스트 청킹
            chunks = self._split_text(text)

            if not chunks:
                raise ValueError("텍스트 청킹 결과가 비어있습니다.")

            # 컬렉션 생성 또는 가져오기
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
            except:
                pass  # 컬렉션이 없는 경우 무시

            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )

            # 청크들을 벡터 DB에 추가
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"chunk_id": i, "length": len(chunk)} for i, chunk in enumerate(chunks)]

            self.collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )

            print(f"✅ 문서 처리 완료: {len(chunks)}개 청크 생성")
            return True

        except Exception as e:
            print(f"❌ 문서 처리 실패: {str(e)}")
            return False

    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[str]:
        """
        쿼리와 관련된 텍스트 청크 검색

        Args:
            query: 검색 쿼리
            k: 반환할 청크 수

        Returns:
            List[str]: 관련 텍스트 청크들
        """
        if not self.collection:
            raise ValueError("문서가 처리되지 않았습니다. process_document()를 먼저 실행하세요.")

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(k, self.collection.count())
            )

            return results['documents'][0] if results['documents'] else []

        except Exception as e:
            print(f"❌ 검색 실패: {str(e)}")
            return []

    def get_document_summary(self) -> str:
        """
        문서 전체 요약 반환

        Returns:
            str: 문서 요약
        """
        return self.document_summary

    def get_all_chunks(self) -> List[str]:
        """
        저장된 모든 청크 반환

        Returns:
            List[str]: 모든 텍스트 청크들
        """
        if not self.collection:
            return []

        try:
            results = self.collection.get()
            return results['documents'] if results['documents'] else []
        except:
            return []

    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        # 기본적인 정제
        text = text.strip()

        # 여러 개의 공백을 하나로 변경
        import re
        text = re.sub(r'\s+', ' ', text)

        # 여러 개의 줄바꿈을 두 개로 제한
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text

    def _create_summary(self, text: str) -> str:
        """문서 요약 생성"""
        # 간단한 요약 (처음 500자)
        summary = text[:500]
        if len(text) > 500:
            summary += "..."

        return summary

    def _split_text(self, text: str) -> List[str]:
        """텍스트를 청크로 분할"""
        chunks = []

        # 문단 단위로 먼저 분할
        paragraphs = text.split('\n\n')

        current_chunk = ""
        for paragraph in paragraphs:
            # 현재 청크에 추가했을 때 크기 확인
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph

            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # 현재 청크를 저장하고 새로운 청크 시작
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # 문단이 너무 큰 경우 더 작게 분할
                if len(paragraph) > self.chunk_size:
                    # 문장 단위로 분할
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    current_chunk = ""

                    for sentence in sentences:
                        test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                        if len(test_chunk) <= self.chunk_size:
                            current_chunk = test_chunk
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence
                else:
                    current_chunk = paragraph

        # 마지막 청크 추가
        if current_chunk:
            chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if len(chunk.strip()) > 50]  # 너무 작은 청크 제외

    def count_tokens(self, text: str) -> int:
        """토큰 수 계산"""
        return len(self.encoding.encode(text))

    def get_collection_info(self) -> dict:
        """컬렉션 정보 반환"""
        if not self.collection:
            return {"status": "no_collection"}

        try:
            count = self.collection.count()
            return {
                "status": "active",
                "collection_name": self.collection_name,
                "total_chunks": count,
                "summary": self.document_summary[:100] + "..." if len(self.document_summary) > 100 else self.document_summary
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}