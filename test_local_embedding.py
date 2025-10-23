"""
Test script for Local Sentence Transformers embeddings
"""
import sys
sys.path.insert(0, 'ai-services')

from src.rag.embeddings import EmbeddingsManager

def main():
    print('=== Local Embedding Test (Sentence Transformers) ===\n')
    
    # 로컬 한국어 모델 사용 (무료, 할당량 없음)
    embeddings = EmbeddingsManager(
        model_name="jhgan/ko-sroberta-multitask",  # 한국어 최적화 모델
        provider="local"
    )

    # Test with a simple text
    test_text = '수학은 논리적 사고를 기르는 중요한 과목입니다.'
    print(f'Test text: {test_text}')

    try:
        print('\n[1] Single embedding test...')
        embedding = embeddings.generate_single_embedding(test_text)
        print(f'✅ Embedding generated successfully!')
        print(f'   Embedding dimension: {len(embedding)}')
        print(f'   First 10 values: {[round(v, 4) for v in embedding[:10]]}')
        
        # Test batch embeddings
        test_texts = [
            '수학은 논리적 사고를 기르는 중요한 과목입니다.',
            '과학은 자연 현상을 이해하는 학문입니다.',
            '역사는 과거를 배우고 미래를 준비합니다.',
            '영어는 국제 의사소통의 중요한 도구입니다.',
            '체육은 건강한 신체를 만드는 과목입니다.'
        ]
        
        print(f'\n[2] Batch embedding test with {len(test_texts)} texts...')
        batch_embeddings = embeddings.generate_embeddings(test_texts)
        print(f'✅ Batch embeddings generated successfully!')
        print(f'   Number of embeddings: {len(batch_embeddings)}')
        print(f'   Each embedding dimension: {len(batch_embeddings[0])}')
        
        # Show cost estimation
        cost_info = embeddings.estimate_cost(test_texts)
        print(f'\n💰 Cost estimation:')
        print(f'   Total tokens: {cost_info["total_tokens"]}')
        print(f'   Total cost: ${cost_info["total_cost"]:.6f} (로컬 모델 = 무료)')
        print(f'   Model: {cost_info.get("model", "local")}')
        
        print(f'\n✨ 로컬 임베딩 모델의 장점:')
        print(f'   - 무료 (API 비용 없음)')
        print(f'   - 할당량 제한 없음')
        print(f'   - 빠른 응답 속도')
        print(f'   - 오프라인 작동 가능')
        print(f'   - 데이터 프라이버시 보장')
        
    except Exception as e:
        print(f'\n❌ Error: {str(e)}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
