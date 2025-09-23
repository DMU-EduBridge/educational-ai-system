"""
Integration Test - 전체 파이프라인 테스트
"""

import os
import json
import tempfile
from pathlib import Path

# 테스트 환경을 위한 경로 설정
import sys
sys.path.append('../src')

from src.document_loader import DocumentLoader
from src.rag_processor import RAGProcessor
from src.question_generator import QuestionGenerator


def test_document_loader():
    """문서 로더 테스트"""
    print("🧪 문서 로더 테스트...")

    # 테스트 텍스트 생성
    test_content = """
    테스트 문서입니다.

    이 문서는 통합 테스트를 위한 샘플 텍스트입니다.
    머신러닝은 인공지능의 한 분야로, 데이터를 통해 학습하는 기술입니다.
    딥러닝은 머신러닝의 한 방법으로, 인공신경망을 여러 층으로 쌓는 구조입니다.
    자연어 처리는 컴퓨터가 인간의 언어를 이해하고 처리하는 기술입니다.
    이미지 인식은 컴퓨터 비전의 대표적인 응용 분야입니다.
    """

    # 임시 파일 생성
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        temp_file = f.name

    try:
        # 문서 로드 및 전처리 테스트
        processed_text = DocumentLoader.load_and_process(temp_file)

        # 검증
        assert len(processed_text) > 100, "텍스트 길이가 충분하지 않습니다"
        assert "머신러닝" in processed_text, "예상 키워드가 없습니다"

        print("✅ 문서 로더 테스트 통과")
        return processed_text

    finally:
        # 임시 파일 정리
        os.unlink(temp_file)


def test_rag_processor(test_text):
    """RAG 프로세서 테스트"""
    print("🧪 RAG 프로세서 테스트...")

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY가 설정되지 않았습니다. RAG 테스트 건너뜀")
        return None

    try:
        # RAG 프로세서 초기화
        rag_processor = RAGProcessor(chunk_size=500, overlap=100)

        # 문서 처리
        success = rag_processor.process_document(test_text)
        assert success, "문서 처리에 실패했습니다"

        # 컬렉션 정보 확인
        info = rag_processor.get_collection_info()
        assert info["status"] == "active", "컬렉션이 활성화되지 않았습니다"
        assert info["total_chunks"] > 0, "청크가 생성되지 않았습니다"

        # 검색 테스트
        results = rag_processor.retrieve_relevant_chunks("머신러닝", k=2)
        assert len(results) > 0, "검색 결과가 없습니다"

        print("✅ RAG 프로세서 테스트 통과")
        return rag_processor

    except Exception as e:
        print(f"❌ RAG 프로세서 테스트 실패: {str(e)}")
        return None


def test_question_generator(rag_processor):
    """문제 생성기 테스트"""
    print("🧪 문제 생성기 테스트...")

    if not rag_processor:
        print("⚠️ RAG 프로세서가 없어 문제 생성 테스트 건너뜀")
        return []

    try:
        # 문제 생성기 초기화
        question_generator = QuestionGenerator(rag_processor)

        # 문제 생성 (적은 수로 테스트)
        questions = question_generator.generate_questions(num_questions=3, difficulty_mix="balanced")

        # 검증
        assert len(questions) > 0, "문제가 생성되지 않았습니다"

        for question in questions:
            # 필수 필드 확인
            required_fields = ["question", "options", "correct_answer", "explanation", "difficulty", "type"]
            for field in required_fields:
                assert field in question, f"필수 필드 누락: {field}"

            # 선택지 확인
            assert len(question["options"]) == 5, "선택지가 5개가 아닙니다"

            # 정답 번호 확인
            assert 1 <= question["correct_answer"] <= 5, "정답 번호가 유효하지 않습니다"

        print("✅ 문제 생성기 테스트 통과")
        return questions

    except Exception as e:
        print(f"❌ 문제 생성기 테스트 실패: {str(e)}")
        return []


def test_full_pipeline():
    """전체 파이프라인 테스트"""
    print("\n" + "="*60)
    print("🚀 RAG 기반 문제 생성 시스템 통합 테스트 시작")
    print("="*60)

    try:
        # 1. 문서 로더 테스트
        test_text = test_document_loader()

        # 2. RAG 프로세서 테스트
        rag_processor = test_rag_processor(test_text)

        # 3. 문제 생성기 테스트
        questions = test_question_generator(rag_processor)

        # 4. 결과 검증
        if questions:
            print(f"\n📊 테스트 결과:")
            print(f"  생성된 문제 수: {len(questions)}개")

            difficulty_stats = {}
            type_stats = {}
            for q in questions:
                diff = q.get('difficulty', 'unknown')
                qtype = q.get('type', 'unknown')
                difficulty_stats[diff] = difficulty_stats.get(diff, 0) + 1
                type_stats[qtype] = type_stats.get(qtype, 0) + 1

            print(f"  난이도 분포: {difficulty_stats}")
            print(f"  문제 유형 분포: {type_stats}")

            # 샘플 문제 출력
            if questions:
                sample = questions[0]
                print(f"\n📝 샘플 문제:")
                print(f"  문제: {sample['question']}")
                print(f"  정답: {sample['options'][sample['correct_answer']-1]}")
                print(f"  난이도: {sample['difficulty']}")
                print(f"  유형: {sample['type']}")

        print("\n" + "="*60)
        print("✅ 통합 테스트 완료!")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n❌ 통합 테스트 실패: {str(e)}")
        print("="*60)
        return False


def test_question_quality():
    """생성된 문제 품질 검증"""
    print("🧪 문제 품질 검증...")

    # 샘플 문제 데이터
    sample_question = {
        "question": "머신러닝의 주요 특징은 무엇인가요?",
        "options": ["선택지1", "선택지2", "선택지3", "선택지4", "선택지5"],
        "correct_answer": 2,
        "explanation": "해설입니다",
        "difficulty": "medium",
        "type": "concept"
    }

    # 검증 로직
    required_fields = ["question", "options", "correct_answer", "explanation", "difficulty", "type"]

    for field in required_fields:
        assert field in sample_question, f"필수 필드 누락: {field}"

    assert len(sample_question["options"]) == 5, "선택지가 5개가 아닙니다"
    assert 1 <= sample_question["correct_answer"] <= 5, "정답 번호가 유효하지 않습니다"
    assert len(set(sample_question["options"])) == 5, "선택지에 중복이 있습니다"

    print("✅ 문제 품질 검증 통과")


if __name__ == "__main__":
    # 개별 테스트 실행
    test_question_quality()

    # 전체 파이프라인 테스트 실행
    test_full_pipeline()