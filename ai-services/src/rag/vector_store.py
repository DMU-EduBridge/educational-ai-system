from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings
import logging
import uuid
from pathlib import Path

from .document_processor import Document


class VectorStore:
    """ChromaDB 기반 벡터 저장소"""

    def __init__(self,
                 collection_name: str = "textbook_embeddings",
                 persist_directory: str = "./data/vector_db"):
        """
        VectorStore 초기화

        Args:
            collection_name: ChromaDB 컬렉션 이름
            persist_directory: 데이터 저장 경로
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.logger = logging.getLogger(__name__)

        # 저장 디렉토리 생성
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # 컬렉션 가져오기 또는 생성
        try:
            self.collection = self.client.get_collection(name=collection_name)
            self.logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Educational textbook embeddings for RAG"}
            )
            self.logger.info(f"Created new collection: {collection_name}")

    def add_documents(self,
                     documents: List[Document],
                     embeddings: List[List[float]]) -> bool:
        """
        문서와 임베딩을 벡터 저장소에 추가

        Args:
            documents: Document 객체 리스트
            embeddings: 임베딩 벡터 리스트

        Returns:
            bool: 성공 여부
        """
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have the same length")

        # 빈 리스트인 경우 성공으로 처리
        if len(documents) == 0:
            return True

        try:
            ids = []
            contents = []
            metadatas = []

            for doc, embedding in zip(documents, embeddings):
                # 고유 ID 생성
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)
                contents.append(doc.content)

                # 메타데이터 준비 (ChromaDB는 중첩된 딕셔너리를 지원하지 않음)
                metadata = {}
                for key, value in doc.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    else:
                        metadata[key] = str(value)

                metadatas.append(metadata)

            # ChromaDB에 추가
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas
            )

            self.logger.info(f"Successfully added {len(documents)} documents to collection")
            return True

        except Exception as e:
            self.logger.error(f"Error adding documents to vector store: {str(e)}")
            raise

    def similarity_search(self,
                         query: str,
                         k: int = 5,
                         filter_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        쿼리에 대한 유사도 검색 수행

        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            filter_metadata: 메타데이터 필터

        Returns:
            List[Document]: 검색 결과 Document 리스트
        """
        try:
            # 임베딩 매니저가 필요하므로 쿼리 임베딩은 외부에서 생성해야 함
            # 이는 RAGRetriever에서 처리됩니다
            raise NotImplementedError(
                "similarity_search requires query embedding. Use RAGRetriever.retrieve_context instead."
            )

        except Exception as e:
            self.logger.error(f"Error in similarity search: {str(e)}")
            raise

    def similarity_search_by_embedding(self,
                                     query_embedding: List[float],
                                     k: int = 5,
                                     filter_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        임베딩 벡터로 유사도 검색 수행

        Args:
            query_embedding: 쿼리 임베딩 벡터
            k: 반환할 결과 수
            filter_metadata: 메타데이터 필터

        Returns:
            List[Document]: 검색 결과 Document 리스트
        """
        try:
            # 필터 조건 준비
            where_clause = None
            if filter_metadata:
                conditions = []
                for key, value in filter_metadata.items():
                    if value is not None:
                        conditions.append({key: value})

                if len(conditions) == 1:
                    where_clause = conditions[0]
                elif len(conditions) > 1:
                    where_clause = {"$and": conditions}

            # ChromaDB 검색
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )

            # 결과를 Document 객체로 변환
            documents = []
            if results['documents'] and results['documents'][0]:
                for i, (content, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # 거리 정보를 메타데이터에 추가
                    metadata['similarity_score'] = 1 - distance  # 거리를 유사도로 변환
                    metadata['distance'] = distance

                    documents.append(Document(
                        content=content,
                        metadata=metadata
                    ))

            self.logger.info(f"Found {len(documents)} documents for query")
            return documents

        except Exception as e:
            self.logger.error(f"Error in similarity search by embedding: {str(e)}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """
        컬렉션 정보 반환

        Returns:
            dict: 컬렉션 통계 정보
        """
        try:
            count = self.collection.count()

            # 메타데이터 분석
            if count > 0:
                # 전체 문서 조회 (최대 1000개)
                sample_size = min(count, 1000)
                results = self.collection.get(limit=sample_size, include=["metadatas"])

                subjects = set()
                units = set()
                source_files = set()

                if results['metadatas']:
                    for metadata in results['metadatas']:
                        if 'subject' in metadata:
                            subjects.add(metadata['subject'])
                        if 'unit' in metadata:
                            units.add(metadata['unit'])
                        if 'source_file' in metadata:
                            source_files.add(metadata['source_file'])

                return {
                    'collection_name': self.collection_name,
                    'total_documents': count,
                    'subjects': list(subjects),
                    'units': list(units),
                    'source_files': list(source_files),
                    'persist_directory': str(self.persist_directory)
                }
            else:
                return {
                    'collection_name': self.collection_name,
                    'total_documents': 0,
                    'subjects': [],
                    'units': [],
                    'source_files': [],
                    'persist_directory': str(self.persist_directory)
                }

        except Exception as e:
            self.logger.error(f"Error getting collection info: {str(e)}")
            raise

    def clear_collection(self) -> bool:
        """
        컬렉션의 모든 데이터 삭제

        Returns:
            bool: 성공 여부
        """
        try:
            # 기존 컬렉션 삭제
            self.client.delete_collection(name=self.collection_name)

            # 새 컬렉션 생성
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Educational textbook embeddings for RAG"}
            )

            self.logger.info(f"Successfully cleared collection: {self.collection_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error clearing collection: {str(e)}")
            raise

    def delete_by_metadata(self, filter_metadata: Dict[str, Any]) -> int:
        """
        메타데이터 조건에 맞는 문서들 삭제

        Args:
            filter_metadata: 삭제할 문서의 메타데이터 조건

        Returns:
            int: 삭제된 문서 수
        """
        try:
            # 삭제할 문서 조회
            results = self.collection.get(
                where=filter_metadata,
                include=["metadatas"]
            )

            if results['ids']:
                # 문서 삭제
                self.collection.delete(ids=results['ids'])
                deleted_count = len(results['ids'])
                self.logger.info(f"Deleted {deleted_count} documents matching criteria")
                return deleted_count
            else:
                self.logger.info("No documents found matching deletion criteria")
                return 0

        except Exception as e:
            self.logger.error(f"Error deleting documents: {str(e)}")
            raise

    def update_metadata(self, document_id: str, new_metadata: Dict[str, Any]) -> bool:
        """
        특정 문서의 메타데이터 업데이트

        Args:
            document_id: 문서 ID
            new_metadata: 새로운 메타데이터

        Returns:
            bool: 성공 여부
        """
        try:
            # ChromaDB는 직접적인 메타데이터 업데이트를 지원하지 않으므로
            # 문서를 다시 추가하는 방식으로 구현
            self.collection.update(
                ids=[document_id],
                metadatas=[new_metadata]
            )

            self.logger.info(f"Successfully updated metadata for document: {document_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error updating metadata: {str(e)}")
            raise