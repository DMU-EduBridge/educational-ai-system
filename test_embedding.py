"""
Test script for Google Gemini embeddings
"""
import sys
sys.path.insert(0, 'ai-services')

from src.rag.embeddings import EmbeddingsManager
from src.utils.config import get_settings

def main():
    print('=== Embedding Test ===')
    config = get_settings()
    embeddings = EmbeddingsManager(
        model_name=config.embedding_model,
        api_key=config.google_api_key
    )

    # Test with a simple text
    test_text = '수학은 논리적 사고를 기르는 중요한 과목입니다.'
    print(f'\nTest text: {test_text}')

    try:
        embedding = embeddings.generate_single_embedding(test_text)
        print(f'\n✅ Embedding generated successfully!')
        print(f'Embedding dimension: {len(embedding)}')
        print(f'First 10 values: {embedding[:10]}')
        
        # Test batch embeddings
        test_texts = [
            '수학은 논리적 사고를 기르는 중요한 과목입니다.',
            '과학은 자연 현상을 이해하는 학문입니다.',
            '역사는 과거를 배우고 미래를 준비합니다.'
        ]
        print(f'\n\nTesting batch embedding with {len(test_texts)} texts...')
        batch_embeddings = embeddings.generate_embeddings(test_texts)
        print(f'✅ Batch embeddings generated successfully!')
        print(f'Number of embeddings: {len(batch_embeddings)}')
        print(f'Each embedding dimension: {len(batch_embeddings[0])}')
        
        # Show cost estimation
        cost_info = embeddings.estimate_cost(test_texts)
        print(f'\n💰 Cost estimation:')
        print(f'   Total tokens: {cost_info["total_tokens"]}')
        print(f'   Total cost: ${cost_info["total_cost"]:.6f}')
        print(f'   Per token: ${cost_info["per_token"]:.10f}')
        
    except Exception as e:
        print(f'\n❌ Error: {str(e)}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
