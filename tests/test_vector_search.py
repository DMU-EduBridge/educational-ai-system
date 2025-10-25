"""벡터 DB 검색 테스트"""
import sys
from pathlib import Path

# ai-services를 path에 추가
project_root = Path(__file__).parent
ai_services_path = project_root / 'ai-services'
sys.path.insert(0, str(ai_services_path))

from src.utils.config import get_settings
from src.rag.vector_store import VectorStore
from src.rag.embeddings import EmbeddingsManager

print("🔧 설정 로드 중...")
settings = get_settings()
print(f"  ChromaDB 경로: {settings.chroma_db_path}")
print(f"  컬렉션 이름: {settings.chroma_collection_name}")

print("\n📦 Embeddings 초기화 중...")
embeddings_manager = EmbeddingsManager(
    provider=settings.embedding_provider,
    model_name=settings.embedding_model
)

print("\n📚 VectorStore 초기화 중...")
vector_store = VectorStore(
    collection_name=settings.chroma_collection_name,
    persist_directory=settings.chroma_db_path
)

print(f"\n📊 벡터 스토어 문서 수: {vector_store.collection.count()}")

# 검색 테스트 1: 필터 사용
query = "이차방정식"
print(f"\n🔍 검색 1: '{query}' (필터: subject='수학', unit='통합교과서')")

# 쿼리 임베딩 생성
query_embedding = embeddings_manager.generate_single_embedding(query)

filter_metadata = {
    "subject": "수학",
    "unit": "통합교과서"
}

results = vector_store.similarity_search_by_embedding(
    query_embedding=query_embedding,
    k=5,
    filter_metadata=filter_metadata
)

print(f"✅ 검색 결과: {len(results)}개 문서")
for i, doc in enumerate(results, 1):
    print(f"\n  문서 {i}:")
    print(f"    메타데이터: {doc.metadata}")
    print(f"    내용: {doc.content[:150]}...")

# 검색 테스트 2: 필터 없이
print(f"\n\n🔍 검색 2: '{query}' (필터 없음)")
results_no_filter = vector_store.similarity_search_by_embedding(
    query_embedding=query_embedding,
    k=5
)

print(f"✅ 검색 결과: {len(results_no_filter)}개 문서")
for i, doc in enumerate(results_no_filter, 1):
    print(f"\n  문서 {i}:")
    print(f"    메타데이터: {doc.metadata}")
    print(f"    내용: {doc.content[:150]}...")
