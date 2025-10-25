"""벡터 DB 내용 확인 스크립트"""
import chromadb
from pathlib import Path

# 벡터 DB 경로
db_path = "/Users/hyunjong_kim/Desktop/KHJ/dongyang/2025_2nd/graduate_project/educational-ai-system/ai-services/data/vector_db"

print(f"📂 ChromaDB 경로: {db_path}")
print(f"📂 경로 존재: {Path(db_path).exists()}")

# ChromaDB 클라이언트 생성
client = chromadb.PersistentClient(path=db_path)

# 컬렉션 목록 확인
collections = client.list_collections()
print(f"\n📚 컬렉션 목록 ({len(collections)}개):")
for col in collections:
    print(f"  - {col.name}")

# textbook_embeddings 컬렉션 확인
if collections:
    collection = client.get_collection("textbook_embeddings")
    count = collection.count()
    print(f"\n📊 textbook_embeddings 컬렉션:")
    print(f"  - 문서 개수: {count}")
    
    if count > 0:
        # 샘플 문서 확인
        results = collection.get(limit=3, include=["documents", "metadatas"])
        print(f"\n📄 샘플 문서:")
        for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
            print(f"\n  문서 {i+1}:")
            print(f"    메타데이터: {meta}")
            print(f"    내용: {doc[:200]}...")
