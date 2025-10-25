"""모델 사전 다운로드 스크립트"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("📥 Downloading embedding model...")
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
print("✅ Embedding model downloaded")

print("\n📥 Downloading reranker model...")
from sentence_transformers import CrossEncoder
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
print("✅ Reranker model downloaded")

print("\n✅ All models downloaded successfully!")
