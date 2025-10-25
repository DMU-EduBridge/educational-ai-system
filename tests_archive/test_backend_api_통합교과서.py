"""백엔드 API 테스트 (통합교과서 사용)"""
import requests
import json

# API 엔드포인트
BASE_URL = "http://localhost:8000"

def test_generate_question():
    """문제 생성 API 테스트"""
    print("📝 문제 생성 API 호출...")
    
    url = f"{BASE_URL}/generate-question"
    data = {
        "subject": "수학",
        "unit": "통합교과서",  # OCR 스크립트가 사용하는 단원명
        "difficulty": "medium",
        "count": 1
    }
    
    print(f"URL: {url}")
    print(f"Data: {json.dumps(data, indent=2, ensure_ascii=False)}")
    print("\n요청 중...\n")
    
    try:
        response = requests.post(url, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            # 응답이 리스트인지 딕셔너리인지 확인
            if isinstance(result, list):
                questions = result
            else:
                questions = result.get('questions', [])
                
            print("✅ 성공!")
            print(f"\n생성된 문제 수: {len(questions)}")
            
            for i, question in enumerate(questions, 1):
                print(f"\n{'='*60}")
                print(f"문제 {i}")
                print(f"{'='*60}")
                print(f"\n제목: {question.get('title', 'N/A')}")
                print(f"\n설명: {question.get('description', 'N/A')}")
                print(f"\n문제: {question.get('content', 'N/A')}")
                print(f"\n선택지:")
                for j, opt in enumerate(question.get('options', []), 1):
                    print(f"  {j}. {opt}")
                print(f"\n정답: {question.get('correct_answer', 'N/A')}")
                print(f"\n해설: {question.get('explanation', 'N/A')}")
                print(f"\n힌트: {', '.join(question.get('hints', []))}")
                print(f"\n태그: {', '.join(question.get('tags', []))}")
        else:
            print(f"❌ 오류 발생 (HTTP {response.status_code})")
            print(f"응답: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 서버 연결 실패. 백엔드 서버가 실행 중인지 확인하세요.")
    except Exception as e:
        print(f"❌ 예외 발생: {str(e)}")

if __name__ == "__main__":
    test_generate_question()
