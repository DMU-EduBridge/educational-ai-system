#!/usr/bin/env python3
"""
QuestionGenerator 배치 생성 테스트 스크립트
"""

import sys
import os
import json
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from unittest.mock import Mock
from src.models.question_generator import QuestionGenerator
from src.models.llm_client import LLMClient
from src.rag.retriever import RAGRetriever


def test_batch_generation():
    """배치 문제 생성 테스트"""
    print("🔄 배치 문제 생성 테스트 시작...")
    
    # Mock 객체들 생성
    mock_llm_client = Mock(spec=LLMClient)
    mock_retriever = Mock(spec=RAGRetriever)
    
    # QuestionGenerator 인스턴스 생성
    generator = QuestionGenerator(
        llm_client=mock_llm_client,
        retriever=mock_retriever
    )
    
    # Mock 설정
    mock_retriever.retrieve_context.return_value = [
        "이차함수는 y = ax² + bx + c 형태입니다.",
        "포물선의 꼭짓점과 대칭축을 구할 수 있습니다."
    ]
    
    # 다양한 문제 응답 준비
    sample_responses = [
        {
            "question": "이차함수 y = x² - 4x + 3에서 꼭짓점의 x좌표는?",
            "options": ["1", "2", "3", "4", "-2"],
            "correct_answer": 2,
            "explanation": "이차함수의 꼭짓점의 x좌표는 -b/2a로 구할 수 있습니다. -(-4)/(2×1) = 2",
            "hint": "이차함수의 꼭짓점 공식 x = -b/2a를 사용하세요.",
            "difficulty": "medium",
            "subject": "수학",
            "unit": "이차함수"
        },
        {
            "question": "이차함수 y = 2x² + 8x + 6의 최솟값은?",
            "options": ["-2", "-4", "2", "6", "0"],
            "correct_answer": 1,
            "explanation": "완전제곱식으로 만들면 y = 2(x+2)² - 2이므로 최솟값은 -2입니다.",
            "hint": "완전제곱식으로 변형해보세요.",
            "difficulty": "medium",
            "subject": "수학",
            "unit": "이차함수"
        },
        {
            "question": "포물선 y = -x² + 6x - 5의 대칭축의 방정식은?",
            "options": ["x = 2", "x = 3", "x = -3", "x = 5", "x = 1"],
            "correct_answer": 2,
            "explanation": "대칭축의 방정식은 x = -b/2a = -6/(2×(-1)) = 3입니다.",
            "hint": "대칭축은 x = -b/2a 공식으로 구할 수 있습니다.",
            "difficulty": "medium",
            "subject": "수학",
            "unit": "이차함수"
        }
    ]
    
    # Mock에서 순차적으로 다른 응답 반환하도록 설정
    mock_llm_client.generate_structured_response.side_effect = sample_responses
    
    print(f"📝 3개의 문제를 배치 생성합니다...")
    
    # 배치 문제 생성
    try:
        results = generator.generate_batch_questions(
            subject="수학",
            unit="이차함수",
            count=3,
            difficulty="medium"
        )
        
        print(f"✅ 배치 생성 완료! {len(results)}개 문제 생성됨")
        
        # 결과 출력
        for i, result in enumerate(results, 1):
            print(f"\n" + "="*50)
            print(f"📝 문제 {i}")
            print("="*50)
            print(f"질문: {result['question']}")
            print(f"정답: {result['correct_answer']}번 - {result['options'][result['correct_answer']-1]}")
            if result.get('hint'):
                print(f"힌트: {result['hint']}")
            print(f"해설: {result['explanation']}")
        
        # 통계 출력
        stats = generator.get_question_statistics()
        print(f"\n📊 최종 통계:")
        print(f"  - 총 문제 수: {stats['total_questions']}개")
        print(f"  - 과목별: {stats['by_subject']}")
        print(f"  - 난이도별: {stats['by_difficulty']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 배치 생성 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_batch_generation()
    sys.exit(0 if success else 1)