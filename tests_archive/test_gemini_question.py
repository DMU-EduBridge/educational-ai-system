"""Gemini API 직접 테스트 - 문제 생성"""
import os
from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# 모델 초기화
model = genai.GenerativeModel(
    model_name='gemini-2.5-flash',
    generation_config={
        'temperature': 1.0,
        'max_output_tokens': 2048,  # 토큰 수 증가
    }
)

# 간단한 테스트 컨텍스트
context = """
제곱근의 뜻과 성질
예를 들어, 7²=49이므로 7은 49의 제곱근이다.
일반적으로 양수의 제곱근은 양수와 음수 2개가 있고, 그 두 수의 절댓값은 서로 같다.
"""

prompt = f"""당신은 중학교 수학 교사입니다. 다음 교과서 내용을 바탕으로 5지선다 문제를 생성하세요.

교과서 내용:
{context}

요구사항:
- 난이도: medium (개념 적용 및 계산)
- 5개의 선택지 (정답 1개, 오답 4개)
- 명확한 해설 포함

**JSON 형식으로만 응답하세요:**

{{
    "question": "문제 본문",
    "options": ["선택지1", "선택지2", "선택지3", "선택지4", "선택지5"],
    "correct_answer": 정답번호(1-5),
    "explanation": "해설"
}}
"""

print("📝 프롬프트 전송 중...")
print(f"프롬프트 길이: {len(prompt)} 문자")

response = model.generate_content(prompt)

print(f"\n✅ 응답 받음")
print(f"\nFinish Reason: {response.candidates[0].finish_reason if response.candidates else 'N/A'}")
print(f"Safety Ratings: {response.candidates[0].safety_ratings if response.candidates else 'N/A'}")

# 응답 텍스트 확인
try:
    print(f"응답 길이: {len(response.text)} 문자")
    print(f"\n응답 내용:")
    print("=" * 80)
    print(response.text)
    print("=" * 80)

    # JSON 파싱 테스트
    import json
    import re
    
    def clean_json(text):
        """JSON 코드 블록 정리"""
        text = text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()
        
        # { ... } 패턴 찾기
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        return text
    
    try:
        cleaned_text = clean_json(response.text)
        print(f"\n정리된 JSON (첫 200자):")
        print(cleaned_text[:200])
        
        data = json.loads(cleaned_text)
        print("\n✅ JSON 파싱 성공!")
        print(f"문제: {data.get('question', 'N/A')}")
        print(f"정답: {data.get('correct_answer', 'N/A')}")
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON 파싱 실패: {e}")
except ValueError as e:
    print(f"\n❌ 응답 텍스트 접근 실패: {e}")
    print(f"\nPrompt Feedback: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'}")
