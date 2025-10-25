"""JSON cleaning 테스트"""
import re

response_text = """```json
{
    "question": "테스트 문제",
    "options": ["1", "2", "3", "4", "5"],
    "correct_answer": 1,
    "explanation": "테스트 설명"
}
```"""

print("원본 응답:")
print(response_text)
print("\n" + "="*80 + "\n")

def _clean_json_response(response: str) -> str:
    """JSON 응답 정리"""
    # 코드 블록 제거
    response = response.strip()
    
    # ```json ... ``` 형식 제거
    if response.startswith('```json'):
        response = response[7:]
    if response.startswith('```'):
        response = response[3:]
    if response.endswith('```'):
        response = response[:-3]

    # 앞뒤 공백 제거
    response = response.strip()
    
    # JSON 객체/배열 추출 시도
    import re
    
    # { ... } 패턴 찾기
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    # [ ... ] 패턴 찾기
    json_match = re.search(r'\[.*\]', response, re.DOTALL)
    if json_match:
        return json_match.group(0)

    return response

cleaned = _clean_json_response(response_text)
print("정리된 응답:")
print(cleaned)
print("\n" + "="*80 + "\n")

# JSON 파싱 테스트
import json
try:
    data = json.loads(cleaned)
    print("✅ JSON 파싱 성공!")
    print(f"문제: {data['question']}")
except json.JSONDecodeError as e:
    print(f"❌ JSON 파싱 실패: {e}")
