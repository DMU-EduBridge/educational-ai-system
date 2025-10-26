#!/usr/bin/env python3
"""
학습 리포트 생성 API 테스트
"""
import requests
import json

# API 엔드포인트
url = "http://localhost:8001/generate-report"

# 요청 데이터 (실제 학생 ID 사용)
data = {
    "user_id": "cmgp37il10002eg3ztaonfui8"  # 정현 학생
}

print("=" * 60)
print("📊 학습 리포트 생성 API 테스트")
print("=" * 60)
print(f"URL: {url}")
print(f"요청 데이터: {json.dumps(data, ensure_ascii=False, indent=2)}")
print()
print("요청 중...")
print()

try:
    response = requests.post(url, json=data, timeout=120)
    
    if response.status_code == 200:
        report = response.json()
        print("✅ 성공! 리포트가 생성되었습니다.")
        print()
        print("=" * 60)
        print("📋 생성된 리포트")
        print("=" * 60)
        print()
        
        print(f"👤 학생 ID: {report.get('user_id')}")
        print(f"⏰ 생성 시간: {report.get('generated_at')}")
        print()
        
        print(f"📌 취약 단원: {report.get('weakest_unit')}")
        print()
        
        print("📊 성적 요약:")
        summary = report.get('performance_summary', {})
        print(f"  - 총 문제 수: {summary.get('total_problems_solved')}")
        print(f"  - 전체 정답률: {summary.get('overall_correct_rate')}")
        print(f"  - 평균 소요 시간: {summary.get('average_time_spent_seconds')}초")
        print()
        
        print("📝 리포트 내용:")
        print("-" * 60)
        print(report.get('report_text', 'N/A'))
        print("-" * 60)
        
    elif response.status_code == 404:
        print(f"❌ 오류 (HTTP 404): 학생을 찾을 수 없거나 학습 데이터가 없습니다.")
        print(f"응답: {response.json()}")
        
    elif response.status_code == 503:
        print(f"❌ 오류 (HTTP 503): 서버를 사용할 수 없습니다.")
        print(f"응답: {response.json()}")
        
    else:
        print(f"❌ 오류 (HTTP {response.status_code})")
        print(f"응답: {response.text}")
        
except requests.exceptions.Timeout:
    print("❌ 타임아웃: 요청 시간이 너무 오래 걸립니다. (120초 초과)")
    
except requests.exceptions.ConnectionError:
    print("❌ 연결 오류: 서버에 연결할 수 없습니다.")
    print("백엔드 서버가 실행 중인지 확인하세요:")
    print("  cd backend && uvicorn main:app --reload --port 8001")
    
except Exception as e:
    print(f"❌ 예상치 못한 오류: {str(e)}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("테스트 완료")
print("=" * 60)
