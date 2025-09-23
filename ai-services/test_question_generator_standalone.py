#!/usr/bin/env python3
"""
QuestionGenerator 단독 테스트 스크립트
"""

import sys
import os
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from unittest.mock import Mock
from src.models.question_generator import QuestionGenerator
from src.models.llm_client import LLMClient
from src.rag.retriever import RAGRetriever


def test_question_generator():
    """QuestionGenerator 기본 테스트"""
    print("🧪 QuestionGenerator 기본 테스트 시작...")
    
    # Mock 객체들 생성
    mock_llm_client = Mock(spec=LLMClient)
    mock_retriever = Mock(spec=RAGRetriever)
    
    # QuestionGenerator 인스턴스 생성
    generator = QuestionGenerator(
        llm_client=mock_llm_client,
        retriever=mock_retriever
    )
    
    print("✅ QuestionGenerator 초기화 성공")
    
    # Mock 설정
    mock_retriever.retrieve_context.return_value = [
        "일차함수는 y = ax + b 형태입니다.",
        "기울기 a는 직선의 기울어진 정도를 나타냅니다.",
        "y절편 b는 y축과 만나는 점의 좌표입니다."
    ]
    
    mock_response = {
        "question": "일차함수 y = 2x + 3에서 기울기는 무엇인가?",
        "options": ["1", "2", "3", "-2", "0"],
        "correct_answer": 2,
        "explanation": "일차함수 y = ax + b에서 a가 기울기이므로, y = 2x + 3에서 기울기는 2입니다.",
        "hint": "일차함수의 일반형 y = ax + b를 생각해보세요.",
        "difficulty": "medium",
        "subject": "수학",
        "unit": "일차함수"
    }
    
    mock_llm_client.generate_structured_response.return_value = mock_response
    
    # 문제 생성 실행
    try:
        result = generator.generate_question(
            subject="수학",
            unit="일차함수",
            difficulty="medium"
        )
        
        print("✅ 문제 생성 성공")
        print(f"📝 생성된 문제: {result['question']}")
        print(f"💡 힌트: {result.get('hint', '없음')}")
        print(f"📚 해설: {result['explanation']}")
        
        # hint 필드 확인
        assert "hint" in result, "hint 필드가 없습니다"
        assert result["hint"] == "일차함수의 일반형 y = ax + b를 생각해보세요.", "hint 내용이 다릅니다"
        
        print("✅ hint 필드 검증 성공")
        
    except Exception as e:
        print(f"❌ 문제 생성 실패: {str(e)}")
        return False
    
    # 문제 검증 테스트
    try:
        valid_question = {
            "question": "테스트 문제",
            "options": ["A", "B", "C", "D", "E"],
            "correct_answer": 1,
            "explanation": "테스트 해설",
            "hint": "이것은 힌트입니다.",
            "difficulty": "medium",
            "subject": "수학",
            "unit": "일차함수"
        }
        
        is_valid = generator.validate_question(valid_question)
        assert is_valid, "유효한 문제가 유효하지 않다고 판단됨"
        
        print("✅ 문제 검증 성공")
        
    except Exception as e:
        print(f"❌ 문제 검증 실패: {str(e)}")
        return False
    
    # hint 없는 문제 검증 테스트
    try:
        valid_question_no_hint = {
            "question": "테스트 문제",
            "options": ["A", "B", "C", "D", "E"],
            "correct_answer": 1,
            "explanation": "테스트 해설",
            "difficulty": "medium",
            "subject": "수학",
            "unit": "일차함수"
            # hint 필드 없음
        }
        
        is_valid = generator.validate_question(valid_question_no_hint)
        assert is_valid, "hint가 없는 유효한 문제가 유효하지 않다고 판단됨"
        
        print("✅ hint 없는 문제 검증 성공")
        
    except Exception as e:
        print(f"❌ hint 없는 문제 검증 실패: {str(e)}")
        return False
    
    print("🎉 모든 테스트 통과!")
    return True


if __name__ == "__main__":
    success = test_question_generator()
    sys.exit(0 if success else 1)