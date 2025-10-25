#!/usr/bin/env python3
"""
백엔드 API를 통한 문제 생성 테스트
"""
import requests
import json

# API 엔드포인트
url = "http://localhost:8000/generate-question"

# 요청 데이터
data = {
    "subject": "수학",
    "unit": "이차방정식",
    "difficulty": "medium",
    "count": 1
}

print("📝 문제 생성 API 호출...")
print(f"URL: {url}")
print(f"Data: {json.dumps(data, ensure_ascii=False, indent=2)}")
print("\n요청 중...")

try:
    response = requests.post(url, json=data, timeout=60)
    
    if response.status_code == 200:
        questions = response.json()
        print(f"\n✅ 성공! {len(questions)}개 문제 생성됨\n")
        
        for i, q in enumerate(questions, 1):
            print(f"=== 문제 {i} ===")
            print(f"문제: {q['question']}\n")
            print("선택지:")
            for j, opt in enumerate(q['options'], 1):
                marker = "✓" if j == q['correct_answer'] else " "
                print(f"{marker} {j}. {opt}")
            print(f"\n정답: {q['correct_answer']}번")
            print(f"\n해설: {q['explanation']}")
            print("=" * 50)
    else:
        print(f"\n❌ 오류 발생 (HTTP {response.status_code})")
        print(f"응답: {response.text}")

except requests.exceptions.Timeout:
    print("\n❌ 요청 시간 초과 (60초)")
except requests.exceptions.ConnectionError:
    print("\n❌ 서버 연결 실패. 백엔드 서버가 실행 중인지 확인하세요.")
except Exception as e:
    print(f"\n❌ 오류: {e}")
    import traceback
    traceback.print_exc()
