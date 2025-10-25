#!/usr/bin/env python3
"""
문제 생성 기능 통합 테스트 (임베딩 없이)
"""

import sys
import os
from pathlib import Path
import json

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent / "ai-services"
sys.path.insert(0, str(project_root))

from src.utils.config import get_settings
from src.models.llm_client import LLMClient

def test_question_generation():
    """문제 생성 프롬프트 테스트"""
    print("=" * 60)
    print("문제 생성 기능 테스트")
    print("=" * 60)
    
    try:
        settings = get_settings()
        gemini_config = settings.get_gemini_config()
        
        llm_client = LLMClient(
            model_name=gemini_config['model'],
            api_key=gemini_config['api_key'],
            temperature=0.7,
            max_tokens=2000
        )
        print(f"✓ LLMClient 초기화 성공")
        print()
        
        # 모의 컨텍스트 (실제로는 RAG에서 검색됨)
        mock_context = """
        일차함수
        
        일차함수는 y = ax + b 형태로 나타낼 수 있는 함수입니다.
        여기서 a는 기울기, b는 y절편을 나타냅니다.
        
        기울기 a는 x의 값이 1만큼 증가할 때 y의 값이 얼마나 변하는지를 나타냅니다.
        y절편 b는 그래프가 y축과 만나는 점의 y좌표입니다.
        
        예를 들어, y = 2x + 3에서 기울기는 2이고 y절편은 3입니다.
        """
        
        # 문제 생성 프롬프트
        prompt = f"""당신은 중학교 수학 과목의 전문 교사입니다.
다음 교과서 내용을 바탕으로 medium 난이도의 5지선다 문제를 1개 생성해주세요.

교과서 내용:
{mock_context}

문제 생성 규칙:
1. 교과서 내용에 직접 관련된 문제.
2. 중학교 3학년 수준에 맞는 명확한 문제.
3. 5개의 선택지 (정답 1개, 매력적인 오답 4개).
4. 상세하고 교육적인 해설.
5. 문제 해결에 도움이 되는 힌트 목록 (최소 1개 이상).
6. 문제의 핵심 내용을 담은 간결한 제목.
7. 문제에 대한 부가적인 설명 (description).
8. 관련 개념을 나타내는 태그 목록 (최소 1개 이상).
9. 모든 내용은 한국어로 작성.

난이도 기준 (medium):
개념 적용 및 계산, 예제 문제 응용

출력 형식 (JSON만 출력, 다른 설명 없이 JSON 객체만 반환):
{{
    "title": "문제의 간결한 제목",
    "description": "문제에 대한 부가적인 설명입니다.",
    "content": "여기에 문제의 본문을 작성합니다.",
    "options": ["1번 선택지", "2번 선택지", "3번 선택지", "4번 선택지", "5번 선택지"],
    "correct_answer": 정답_번호(1-5 사이의 숫자),
    "explanation": "정답에 대한 상세하고 친절한 해설입니다.",
    "hints": ["문제 해결에 도움이 되는 첫 번째 힌트"],
    "tags": ["관련_태그_1"]
}}
"""
        
        print("문제 생성 중...")
        print()
        
        response = llm_client.generate_structured_response(
            prompt=prompt,
            response_format="json",
            max_tokens=2000
        )
        
        print("✓ 문제 생성 성공!")
        print()
        print("=" * 60)
        print("생성된 문제")
        print("=" * 60)
        print()
        
        print(f"📌 제목: {response.get('title', 'N/A')}")
        print(f"📝 설명: {response.get('description', 'N/A')}")
        print()
        print(f"❓ 문제:")
        print(f"   {response.get('content', 'N/A')}")
        print()
        
        print("선택지:")
        for i, option in enumerate(response.get('options', []), 1):
            marker = "✓" if i == response.get('correct_answer') else " "
            print(f"  {marker} {i}. {option}")
        print()
        
        print(f"💡 해설:")
        print(f"   {response.get('explanation', 'N/A')}")
        print()
        
        print(f"🔍 힌트:")
        for hint in response.get('hints', []):
            print(f"   - {hint}")
        print()
        
        print(f"🏷️  태그: {', '.join(response.get('tags', []))}")
        print()
        
        # 사용량 통계
        usage = llm_client.track_usage()
        print("=" * 60)
        print(f"사용량: {usage['total_tokens']} 토큰, 비용: ${usage['total_cost_usd']:.6f}")
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
    success = test_question_generation()
    sys.exit(0 if success else 1)
