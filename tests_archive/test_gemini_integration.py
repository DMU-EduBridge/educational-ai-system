#!/usr/bin/env python3
"""
Google Gemini API 통합 테스트 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent / "ai-services"
sys.path.insert(0, str(project_root))

from src.utils.config import get_settings
from src.models.llm_client import LLMClient
from src.rag.embeddings import EmbeddingsManager

def test_llm_client():
    """LLMClient 테스트"""
    print("=" * 60)
    print("1. LLMClient (Google Gemini) 테스트")
    print("=" * 60)
    
    try:
        settings = get_settings()
        gemini_config = settings.get_gemini_config()
        
        print(f"모델: {gemini_config['model']}")
        print(f"온도: {gemini_config['temperature']}")
        print(f"최대 토큰: {gemini_config['max_tokens']}")
        print()
        
        # LLMClient 초기화
        llm_client = LLMClient(
            model_name=gemini_config['model'],
            api_key=gemini_config['api_key'],
            temperature=gemini_config['temperature'],
            max_tokens=gemini_config['max_tokens']
        )
        
        # 간단한 응답 생성 테스트
        test_prompt = "안녕하세요! 간단한 인사를 한국어로 해주세요."
        print(f"프롬프트: {test_prompt}")
        print()
        
        response = llm_client.generate_response(test_prompt)
        print(f"응답: {response}")
        print()
        
        # 사용량 추적
        usage = llm_client.track_usage()
        print(f"사용량 통계:")
        print(f"  총 요청: {usage['total_requests']}")
        print(f"  총 토큰: {usage['total_tokens']}")
        print(f"  총 비용: ${usage['total_cost_usd']:.6f}")
        print()
        
        print("✅ LLMClient 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"❌ LLMClient 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_embeddings():
    """EmbeddingsManager 테스트"""
    print()
    print("=" * 60)
    print("2. EmbeddingsManager (Google Embeddings) 테스트")
    print("=" * 60)
    
    try:
        settings = get_settings()
        gemini_config = settings.get_gemini_config()
        
        print(f"임베딩 모델: {gemini_config['embedding_model']}")
        print()
        
        # EmbeddingsManager 초기화
        embeddings_manager = EmbeddingsManager(
            model_name=gemini_config['embedding_model'],
            api_key=gemini_config['api_key']
        )
        
        # 테스트 텍스트
        test_texts = [
            "일차함수는 y = ax + b 형태의 함수입니다.",
            "이차함수는 y = ax² + bx + c 형태의 함수입니다."
        ]
        
        print(f"테스트 텍스트 ({len(test_texts)}개):")
        for i, text in enumerate(test_texts, 1):
            print(f"  {i}. {text}")
        print()
        
        # 비용 추정
        cost_info = embeddings_manager.estimate_cost(test_texts)
        print(f"비용 추정:")
        print(f"  총 토큰: {cost_info['total_tokens']}")
        print(f"  예상 비용: ${cost_info['estimated_cost_usd']:.6f}")
        print()
        
        # 임베딩 생성
        print("임베딩 생성 중...")
        embeddings = embeddings_manager.generate_embeddings(test_texts)
        
        print(f"생성된 임베딩 수: {len(embeddings)}")
        print(f"임베딩 차원: {len(embeddings[0])}")
        print(f"첫 번째 임베딩 샘플: [{embeddings[0][:5]}...]")
        print()
        
        print("✅ EmbeddingsManager 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"❌ EmbeddingsManager 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_structured_response():
    """구조화된 응답 생성 테스트"""
    print()
    print("=" * 60)
    print("3. 구조화된 응답 (JSON) 생성 테스트")
    print("=" * 60)
    
    try:
        settings = get_settings()
        gemini_config = settings.get_gemini_config()
        
        llm_client = LLMClient(
            model_name=gemini_config['model'],
            api_key=gemini_config['api_key'],
            temperature=0.7,
            max_tokens=1000
        )
        
        prompt = """
        다음 정보를 JSON 형식으로 출력해주세요:
        - 이름: 일차함수
        - 설명: 일차식으로 나타낼 수 있는 함수
        - 예시: y = 2x + 3
        
        JSON만 출력하고 다른 설명은 하지 마세요.
        """
        
        print("프롬프트:")
        print(prompt)
        print()
        
        response = llm_client.generate_structured_response(
            prompt=prompt,
            response_format="json"
        )
        
        print("응답 (JSON):")
        import json
        print(json.dumps(response, indent=2, ensure_ascii=False))
        print()
        
        print("✅ 구조화된 응답 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"❌ 구조화된 응답 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 테스트 함수"""
    print()
    print("🚀 Google Gemini API 통합 테스트 시작")
    print()
    
    # 환경 변수 확인
    try:
        settings = get_settings()
        if not settings.validate_api_key():
            print("❌ 오류: Google API 키가 설정되지 않았거나 유효하지 않습니다.")
            print("   .env 파일에 GOOGLE_API_KEY를 설정해주세요.")
            return False
    except Exception as e:
        print(f"❌ 설정 로드 실패: {str(e)}")
        print("   .env 파일이 올바르게 설정되었는지 확인해주세요.")
        return False
    
    # 테스트 실행
    results = []
    
    results.append(("LLMClient", test_llm_client()))
    results.append(("EmbeddingsManager", test_embeddings()))
    results.append(("구조화된 응답", test_structured_response()))
    
    # 결과 요약
    print()
    print("=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    for name, success in results:
        status = "✅ 성공" if success else "❌ 실패"
        print(f"{name}: {status}")
    
    all_success = all(result[1] for result in results)
    
    print()
    if all_success:
        print("🎉 모든 테스트가 성공적으로 완료되었습니다!")
        print("   Google Gemini API 통합이 정상적으로 작동합니다.")
    else:
        print("⚠️  일부 테스트가 실패했습니다.")
        print("   오류 메시지를 확인하고 문제를 해결해주세요.")
    
    return all_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
