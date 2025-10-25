#!/usr/bin/env python3
"""
백엔드 API를 통해 '이차함수' 단원 문제 생성 테스트
"""

import requests
import json

# 백엔드 API URL
API_URL = "http://localhost:8000/generate-question"

# 요청 데이터
request_data = {
    "subject": "수학",
    "unit": "이차함수",
    "difficulty": "medium",
    "count": 1
}

print("=" * 80)
print(f"백엔드 API 테스트: {API_URL}")
print("=" * 80)
print(f"요청 데이터: {json.dumps(request_data, ensure_ascii=False, indent=2)}")
print("=" * 80)

try:
    # POST 요청
    response = requests.post(API_URL, json=request_data, timeout=120)
    
    print(f"응답 상태 코드: {response.status_code}")
    
    if response.status_code == 200:
        questions = response.json()
        
        print("\n✅ 문제 생성 성공!")
        print("=" * 80)
        
        for i, q in enumerate(questions, 1):
            print(f"\n문제 {i}:")
            print(f"제목: {q.get('title', 'N/A')}")
            print(f"설명: {q.get('description', 'N/A')}")
            print(f"\n내용:\n{q.get('content', 'N/A')}")
            
            print(f"\n선택지:")
            for idx, option in enumerate(q.get('options', []), 1):
                print(f"  {idx}. {option}")
            
            print(f"\n정답: {q.get('correct_answer', 'N/A')}")
            print(f"\n해설:\n{q.get('explanation', 'N/A')}")
            
            print(f"\n힌트:")
            for hint in q.get('hints', []):
                print(f"  - {hint}")
            
            print(f"\n태그: {', '.join(q.get('tags', []))}")
            print("=" * 80)
    else:
        print(f"❌ 오류 발생: {response.status_code}")
        print(f"응답 내용: {response.text}")

except Exception as e:
    print(f"❌ 예외 발생: {str(e)}")
