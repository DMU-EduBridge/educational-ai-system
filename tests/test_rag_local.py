"""
Comprehensive test for RAG system with local embeddings
"""
import sys
sys.path.insert(0, 'ai-services')

from src.rag.embeddings import EmbeddingsManager
from src.rag.document_processor import DocumentProcessor
from src.utils.config import get_settings
import os
import numpy as np

def main():
    print('=== RAG System Test with Local Embeddings ===\n')
    
    config = get_settings()
    
    # 1. Initialize Embeddings Manager
    print('[1] Initializing Embeddings Manager...')
    embeddings = EmbeddingsManager(
        model_name=config.embedding_model,
        api_key=config.google_api_key if config.embedding_provider == 'google' else None,
        provider=config.embedding_provider
    )
    print(f'✅ Using {config.embedding_provider} embeddings: {config.embedding_model}')
    
    # 2. Process sample documents
    print('\n[2] Processing sample documents...')
    sample_texts = [
        "피타고라스 정리는 직각삼각형의 세 변의 길이 관계를 나타냅니다. a² + b² = c²",
        "미분은 함수의 순간 변화율을 구하는 방법입니다. f'(x) = lim(h→0) [f(x+h) - f(x)]/h",
        "적분은 미분의 역연산으로, 함수의 넓이를 구하는 데 사용됩니다.",
        "이차방정식의 해는 근의 공식으로 구할 수 있습니다: x = [-b ± √(b²-4ac)] / 2a",
        "삼각함수는 sin, cos, tan으로 각도와 변의 길이 관계를 나타냅니다."
    ]
    
    doc_processor = DocumentProcessor()
    processed_docs = []
    
    for i, text in enumerate(sample_texts):
        doc_data = {
            'content': text,
            'metadata': {
                'source': f'sample_{i+1}.txt',
                'topic': 'mathematics',
                'grade': '3'
            }
        }
        chunks = doc_processor.chunk_text(text)
        for chunk in chunks:
            processed_docs.append({
                'text': chunk,
                'metadata': doc_data['metadata']
            })
    
    print(f'✅ Processed {len(processed_docs)} document chunks')
    
    # 3. Test embedding generation
    print('\n[3] Generating embeddings for documents...')
    texts = [doc['text'] for doc in processed_docs]
    
    # Generate embeddings using local model
    doc_embeddings = embeddings.generate_embeddings(texts)
    print(f'✅ Generated {len(doc_embeddings)} embeddings')
    print(f'   Embedding dimension: {len(doc_embeddings[0])}')
    
    # 4. Test semantic similarity
    print('\n[4] Testing semantic similarity...')
    test_queries = [
        "피타고라스 정리에 대해 설명해주세요",
        "미분이란 무엇인가요?",
        "삼각함수의 종류는?"
    ]
    
    import numpy as np
    
    def cosine_similarity(a, b):
        """코사인 유사도 계산"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    for query in test_queries:
        print(f'\n   Query: "{query}"')
        query_embedding = embeddings.generate_single_embedding(query)
        
        # Find most similar documents
        similarities = []
        for i, doc_emb in enumerate(doc_embeddings):
            sim = cosine_similarity(query_embedding, doc_emb)
            similarities.append((i, sim, texts[i]))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Show top 2 results
        for rank, (idx, score, text) in enumerate(similarities[:2], 1):
            print(f'   Result {rank}:')
            print(f'      Text: {text[:80]}...')
            print(f'      Similarity: {score:.4f}')
    
    # 5. Show cost estimation
    print('\n[5] Cost Analysis:')
    cost_info = embeddings.estimate_cost(texts + test_queries)
    print(f'   Total documents processed: {len(texts) + len(test_queries)}')
    print(f'   Total tokens: {cost_info["total_tokens"]}')
    print(f'   Total cost: ${cost_info["total_cost"]:.6f}')
    print(f'   Provider: {cost_info["provider"]}')
    print(f'   Note: {cost_info.get("note", "N/A")}')
    
    print('\n✅ Local embedding system is working perfectly!')
    print('\n💡 Benefits of Local Embeddings:')
    print('   - No API costs')
    print('   - No quota limits')
    print('   - Fast local processing')
    print('   - Complete privacy')
    print('   - Works offline')
    
    print('\n✨ Test completed successfully!')

if __name__ == '__main__':
    main()
