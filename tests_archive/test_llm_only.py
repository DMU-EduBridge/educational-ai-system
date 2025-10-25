#!/usr/bin/env python3
"""
Google Gemini LLM만 테스트하는 간단한 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent / "ai-services"
sys.path.insert(0, str(project_root))

from src.utils.config import get_settings
from src.models.llm_client import LLMClient

def test_llm():
    """LLM 기본 기능 테스트"""
    print("=" * 60)
    print("Google Gemini LLM 테스트")
    print("=" * 60)
    
    try:
        settings = get_settings()
        gemini_config = settings.get_gemini_config()
        
        print(f"✓ 설정 로드 성공")
        print(f"  모델: {gemini_config['model']}")
        print(f"  온도: {gemini_config['temperature']}")
        print()
        
        # LLMClient 초기화
        llm_client = LLMClient(
            model_name=gemini_config['model'],
            api_key=gemini_config['api_key'],
            temperature=gemini_config['temperature'],
            max_tokens=gemini_config['max_tokens']
        )
        print(f"✓ LLMClient 초기화 성공")
        print()
        
        # 테스트 1: 간단한 응답
        print("테스트 1: 간단한 응답 생성")
        print("-" * 60)
        test_prompt = "안녕하세요를 영어로 번역해주세요. 번역만 답변해주세요."
        print(f"프롬프트: {test_prompt}")
        
        response = llm_client.generate_response(test_prompt)
        print(f"응답: {response}")
        print()
        
        # 테스트 2: 시스템 메시지 포함
        print("테스트 2: 시스템 메시지 포함")
        print("-" * 60)
        system_msg = "당신은 수학 선생님입니다."
        user_msg = "일차함수가 무엇인지 한 문장으로 설명해주세요."
        print(f"시스템: {system_msg}")
        print(f"사용자: {user_msg}")
        
        response2 = llm_client.generate_response(user_msg, system_message=system_msg)
        print(f"응답: {response2}")
        print()
        
        # 테스트 3: JSON 응답
        print("테스트 3: JSON 응답 생성")
        print("-" * 60)
        json_prompt = """다음을 JSON 형식으로 출력하세요:
        - 이름: 피타고라스 정리
        - 공식: a² + b² = c²
        
        다른 설명 없이 JSON만 출력하세요."""
        print(f"프롬프트: {json_prompt}")
        
        json_response = llm_client.generate_structured_response(json_prompt, response_format="json")
        print(f"응답:")
        import json
        print(json.dumps(json_response, indent=2, ensure_ascii=False))
        print()
        
        # 사용량 통계
        print("사용량 통계")
        print("-" * 60)
        usage = llm_client.track_usage()
        print(f"총 요청: {usage['total_requests']}")
        print(f"총 토큰: {usage['total_tokens']}")
        print(f"총 비용: ${usage['total_cost_usd']:.6f}")
        print()
        
        print("=" * 60)
        print("✅ 모든 테스트 성공!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ 테스트 실패: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_llm()
    sys.exit(0 if success else 1)
