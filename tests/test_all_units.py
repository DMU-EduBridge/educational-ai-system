#!/usr/bin/env python3
"""
여러 단원에 대한 문제 생성 테스트
"""

import requests
import json

API_URL = "http://localhost:8000/generate-question"

# 테스트할 단원 목록
units = [
    "실수와 그 계산",
    "이차방정식",
    "이차함수",
    "삼각비",
    "원의 성질",
    "통계"
]

print("=" * 80)
print("모든 단원별 문제 생성 테스트")
print("=" * 80)

results = {}

for unit in units:
    print(f"\n📚 테스트 중: {unit}")
    print("-" * 80)
    
    request_data = {
        "subject": "수학",
        "unit": unit,
        "difficulty": "medium",
        "count": 1
    }
    
    try:
        response = requests.post(API_URL, json=request_data, timeout=120)
        
        if response.status_code == 200:
            questions = response.json()
            q = questions[0]
            
            print(f"✅ 성공!")
            print(f"제목: {q.get('title', 'N/A')}")
            print(f"태그: {', '.join(q.get('tags', []))}")
            
            results[unit] = "✅ 성공"
        else:
            print(f"❌ 실패: HTTP {response.status_code}")
            results[unit] = f"❌ 실패 ({response.status_code})"
    
    except Exception as e:
        print(f"❌ 예외: {str(e)}")
        results[unit] = f"❌ 예외: {str(e)}"

print("\n" + "=" * 80)
print("테스트 결과 요약")
print("=" * 80)

for unit, result in results.items():
    print(f"{unit}: {result}")

# 성공/실패 통계
success_count = sum(1 for r in results.values() if r.startswith("✅"))
total_count = len(results)

print("\n" + "=" * 80)
print(f"전체: {total_count}개 단원 중 {success_count}개 성공")
print("=" * 80)
