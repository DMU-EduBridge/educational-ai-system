#!/usr/bin/env python3
"""
QuestionGenerator 실제 출력 확인 스크립트
실제 AI 모델 없이 Mock을 사용하여 다양한 문제 생성 결과를 확인합니다.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from unittest.mock import Mock
from src.models.question_generator import QuestionGenerator
from src.models.llm_client import LLMClient
from src.rag.retriever import RAGRetriever


def print_separator(title=""):
    """구분선 출력"""
    print("\n" + "="*60)
    if title:
        print(f"🎯 {title}")
        print("="*60)


def print_question_result(result, title="문제 생성 결과"):
    """문제 결과를 보기 좋게 출력"""
    print_separator(title)
    
    print(f"📝 문제: {result['question']}")
    print(f"📚 과목: {result['subject']} | 단원: {result['unit']} | 난이도: {result['difficulty']}")
    
    print("\n🔤 선택지:")
    for i, option in enumerate(result['options'], 1):
        marker = "✅" if i == result['correct_answer'] else "  "
        print(f"  {marker} {i}. {option}")
    
    print(f"\n📖 해설:")
    print(f"  {result['explanation']}")
    
    if result.get('hint') and result['hint'].strip():
        print(f"\n💡 힌트:")
        print(f"  {result['hint']}")
    else:
        print(f"\n💡 힌트: 제공되지 않음")
    
    print(f"\n📊 메타데이터:")
    print(f"  - ID: {result.get('id', 'N/A')}")
    print(f"  - 생성시간: {result.get('generated_at', 'N/A')}")


def create_sample_questions():
    """다양한 샘플 문제들을 생성"""
    
    # Mock 객체들 생성
    mock_llm_client = Mock(spec=LLMClient)
    mock_retriever = Mock(spec=RAGRetriever)
    
    # QuestionGenerator 인스턴스 생성
    generator = QuestionGenerator(
        llm_client=mock_llm_client,
        retriever=mock_retriever
    )
    
    # 샘플 문제들 정의
    sample_questions = [
        {
            "title": "수학 - 일차함수 (쉬움, 힌트 포함)",
            "context": ["일차함수는 y = ax + b 형태입니다.", "기울기 a는 직선의 기울어진 정도를 나타냅니다."],
            "response": {
                "question": "일차함수 y = 2x + 3에서 기울기는 무엇인가?",
                "options": ["1", "2", "3", "-2", "0"],
                "correct_answer": 2,
                "explanation": "일차함수 y = ax + b에서 a가 기울기이므로, y = 2x + 3에서 기울기는 2입니다.",
                "hint": "일차함수의 일반형 y = ax + b를 생각해보세요.",
                "difficulty": "easy",
                "subject": "수학",
                "unit": "일차함수"
            },
            "params": {"subject": "수학", "unit": "일차함수", "difficulty": "easy"}
        },
        {
            "title": "과학 - 물질의 상태 (보통, 힌트 없음)",
            "context": ["물질은 고체, 액체, 기체의 세 가지 상태로 존재합니다.", "온도와 압력에 따라 상태가 변화합니다."],
            "response": {
                "question": "물이 얼음으로 변하는 과정을 무엇이라고 하는가?",
                "options": ["응고", "융해", "증발", "승화", "응축"],
                "correct_answer": 1,
                "explanation": "물이 얼음으로 변하는 과정은 액체에서 고체로 변하는 것으로 응고라고 합니다.",
                "difficulty": "medium",
                "subject": "과학",
                "unit": "물질의 상태"
            },
            "params": {"subject": "과학", "unit": "물질의 상태", "difficulty": "medium"}
        },
        {
            "title": "국어 - 문법 (어려움, 복합 힌트)",
            "context": ["품사는 단어를 기능과 의미에 따라 분류한 것입니다.", "체언에는 명사, 대명사, 수사가 있습니다."],
            "response": {
                "question": "다음 중 체언이 아닌 것은?",
                "options": ["학교", "그것", "셋", "예쁘다", "하나"],
                "correct_answer": 4,
                "explanation": "'예쁘다'는 형용사로 용언에 해당합니다. 나머지는 모두 체언(명사, 대명사, 수사)입니다.",
                "hint": "체언은 문장에서 주어나 목적어 역할을 할 수 있는 품사입니다.",
                "difficulty": "hard",
                "subject": "국어",
                "unit": "문법"
            },
            "params": {"subject": "국어", "unit": "문법", "difficulty": "hard"}
        },
        {
            "title": "영어 - 시제 (보통, 학습 전략 힌트)",
            "context": ["현재완료는 과거에 시작된 동작이 현재까지 지속되거나 영향을 미칠 때 사용합니다."],
            "response": {
                "question": "다음 중 현재완료 시제가 올바르게 사용된 문장은?",
                "options": [
                    "I have been to Seoul yesterday.",
                    "I have lived here for 5 years.",
                    "I have went to the store.",
                    "I have see the movie last week.",
                    "I have eating lunch now."
                ],
                "correct_answer": 2,
                "explanation": "'I have lived here for 5 years.'가 올바른 현재완료 형태입니다. have + 과거분사 형태로 지속을 나타냅니다.",
                "hint": "현재완료는 'have/has + 과거분사' 형태를 사용하며, for나 since와 함께 쓰입니다.",
                "difficulty": "medium",
                "subject": "영어",
                "unit": "시제"
            },
            "params": {"subject": "영어", "unit": "시제", "difficulty": "medium"}
        }
    ]
    
    print_separator("QuestionGenerator 실제 출력 결과 확인")
    print("📚 다양한 과목과 난이도의 문제 생성 결과를 확인합니다.")
    print("🤖 Mock 데이터를 사용하여 실제 AI 없이도 출력 형식을 확인할 수 있습니다.")
    
    results = []
    
    for i, sample in enumerate(sample_questions, 1):
        print(f"\n🔄 문제 {i}/{len(sample_questions)} 생성 중...")
        
        # Mock 설정
        mock_retriever.retrieve_context.return_value = sample["context"]
        mock_llm_client.generate_structured_response.return_value = sample["response"]
        
        try:
            # 문제 생성
            result = generator.generate_question(**sample["params"])
            results.append(result)
            
            # 결과 출력
            print_question_result(result, sample["title"])
            
        except Exception as e:
            print(f"❌ 문제 {i} 생성 실패: {str(e)}")
    
    return results, generator


def show_statistics(generator):
    """통계 정보 출력"""
    stats = generator.get_question_statistics()
    
    print_separator("생성 통계")
    print(f"📊 총 생성된 문제 수: {stats['total_questions']}개")
    
    print(f"\n📚 과목별 분포:")
    for subject, count in stats['by_subject'].items():
        print(f"  - {subject}: {count}개")
    
    print(f"\n📈 난이도별 분포:")
    for difficulty, count in stats['by_difficulty'].items():
        print(f"  - {difficulty}: {count}개")
    
    print(f"\n📖 단원별 분포:")
    for unit, count in stats['by_unit'].items():
        print(f"  - {unit}: {count}개")


def export_to_json(results, filename="question_output_sample.json"):
    """결과를 JSON 파일로 내보내기"""
    output_data = {
        "generated_at": datetime.now().isoformat(),
        "total_questions": len(results),
        "questions": results
    }
    
    output_path = Path(filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print_separator("JSON 파일 생성")
    print(f"📁 파일 경로: {output_path.absolute()}")
    print(f"📄 파일 크기: {output_path.stat().st_size} bytes")
    print(f"🔢 총 문제 수: {len(results)}개")


def main():
    """메인 실행 함수"""
    try:
        # 문제 생성 및 출력
        results, generator = create_sample_questions()
        
        # 통계 출력
        show_statistics(generator)
        
        # JSON 파일로 내보내기
        export_to_json(results)
        
        print_separator("완료")
        print("🎉 모든 문제 생성 및 출력이 완료되었습니다!")
        print("💡 JSON 파일을 확인하여 구조화된 데이터를 볼 수 있습니다.")
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)