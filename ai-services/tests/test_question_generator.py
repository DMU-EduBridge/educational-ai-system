import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from src.models.question_generator import QuestionGenerator
from src.models.llm_client import LLMClient
from src.rag.retriever import RAGRetriever


class TestQuestionGenerator:
    """QuestionGenerator 테스트 클래스"""

    def setup_method(self):
        """각 테스트 메서드 실행 전 초기화"""
        # Mock 객체들 생성
        self.mock_llm_client = Mock(spec=LLMClient)
        self.mock_retriever = Mock(spec=RAGRetriever)

        # QuestionGenerator 인스턴스 생성
        self.generator = QuestionGenerator(
            llm_client=self.mock_llm_client,
            retriever=self.mock_retriever
        )

    def test_init(self):
        """초기화 테스트"""
        assert self.generator.llm_client == self.mock_llm_client
        assert self.generator.retriever == self.mock_retriever
        assert self.generator.question_history == []

    def test_generate_question_success(self):
        """정상적인 문제 생성 테스트"""
        # Mock 설정
        self.mock_retriever.retrieve_context.return_value = [
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

        self.mock_llm_client.generate_structured_response.return_value = mock_response

        # 문제 생성 실행
        result = self.generator.generate_question(
            subject="수학",
            unit="일차함수",
            difficulty="medium"
        )

        # 검증
        assert result is not None
        assert result["question"] == mock_response["question"]
        assert result["correct_answer"] == mock_response["correct_answer"]
        assert result["difficulty"] == "medium"
        assert result["subject"] == "수학"
        assert result["unit"] == "일차함수"
        assert "generated_at" in result
        assert "id" in result

        # Mock 호출 확인
        self.mock_retriever.retrieve_context.assert_called_once()
        self.mock_llm_client.generate_structured_response.assert_called_once()

        # 히스토리에 추가됐는지 확인
        assert len(self.generator.question_history) == 1

    def test_generate_question_no_context(self):
        """컨텍스트가 없을 때 예외 처리 테스트"""
        # 빈 컨텍스트 반환
        self.mock_retriever.retrieve_context.return_value = []

        with pytest.raises(ValueError) as exc_info:
            self.generator.generate_question("수학", "일차함수", "medium")

        assert "No context found" in str(exc_info.value)

    def test_generate_question_with_custom_query(self):
        """커스텀 쿼리를 사용한 문제 생성 테스트"""
        self.mock_retriever.retrieve_context.return_value = ["테스트 컨텍스트"]

        mock_response = {
            "question": "테스트 문제",
            "options": ["1", "2", "3", "4", "5"],
            "correct_answer": 1,
            "explanation": "테스트 해설",
            "difficulty": "easy",
            "subject": "수학",
            "unit": "일차함수"
        }

        self.mock_llm_client.generate_structured_response.return_value = mock_response

        result = self.generator.generate_question(
            subject="수학",
            unit="일차함수",
            difficulty="easy",
            custom_query="일차함수의 기울기"
        )

        # 커스텀 쿼리가 사용됐는지 확인
        self.mock_retriever.retrieve_context.assert_called_with(
            query="일차함수의 기울기",
            subject="수학",
            unit="일차함수",
            k=3
        )

    def test_generate_batch_questions(self):
        """배치 문제 생성 테스트"""
        # Mock 설정
        self.mock_retriever.retrieve_context.return_value = ["테스트 컨텍스트"]

        mock_response = {
            "question": "테스트 문제",
            "options": ["1", "2", "3", "4", "5"],
            "correct_answer": 1,
            "explanation": "테스트 해설",
            "difficulty": "medium",
            "subject": "수학",
            "unit": "일차함수"
        }

        self.mock_llm_client.generate_structured_response.return_value = mock_response

        # 3개 문제 생성
        results = self.generator.generate_batch_questions(
            subject="수학",
            unit="일차함수",
            count=3,
            difficulty="medium"
        )

        assert len(results) == 3
        assert all(isinstance(q, dict) for q in results)
        assert len(self.generator.question_history) == 3

        # LLM이 3번 호출됐는지 확인
        assert self.mock_llm_client.generate_structured_response.call_count == 3

    def test_generate_batch_questions_with_failures(self):
        """실패가 있는 배치 문제 생성 테스트"""
        # 첫 번째 호출은 실패, 나머지는 성공
        self.mock_retriever.retrieve_context.return_value = ["테스트 컨텍스트"]

        mock_response = {
            "question": "테스트 문제",
            "options": ["1", "2", "3", "4", "5"],
            "correct_answer": 1,
            "explanation": "테스트 해설",
            "difficulty": "medium",
            "subject": "수학",
            "unit": "일차함수"
        }

        # 첫 번째 호출에서 예외 발생, 나머지는 정상
        self.mock_llm_client.generate_structured_response.side_effect = [
            Exception("첫 번째 실패"),
            mock_response,
            mock_response
        ]

        results = self.generator.generate_batch_questions(
            subject="수학",
            unit="일차함수",
            count=3,
            difficulty="medium"
        )

        # 실패한 것 제외하고 2개만 생성되어야 함
        assert len(results) == 2

    def test_validate_question_valid(self):
        """유효한 문제 검증 테스트"""
        valid_question = {
            "question": "일차함수 y = 2x + 3에서 기울기는?",
            "options": ["1", "2", "3", "4", "5"],
            "correct_answer": 2,
            "explanation": "기울기는 2입니다.",
            "difficulty": "medium",
            "subject": "수학",
            "unit": "일차함수"
        }

        result = self.generator.validate_question(valid_question)
        assert result is True

    def test_validate_question_missing_fields(self):
        """필수 필드가 누락된 문제 검증 테스트"""
        invalid_question = {
            "question": "테스트 문제",
            "options": ["1", "2", "3", "4", "5"],
            # correct_answer 누락
            "explanation": "테스트 해설"
        }

        result = self.generator.validate_question(invalid_question)
        assert result is False

    def test_validate_question_invalid_options(self):
        """잘못된 선택지 검증 테스트"""
        invalid_question = {
            "question": "테스트 문제",
            "options": ["1", "2", "3"],  # 3개만 있음 (5개여야 함)
            "correct_answer": 1,
            "explanation": "테스트 해설",
            "difficulty": "medium",
            "subject": "수학",
            "unit": "일차함수"
        }

        result = self.generator.validate_question(invalid_question)
        assert result is False

    def test_validate_question_invalid_correct_answer(self):
        """잘못된 정답 번호 검증 테스트"""
        invalid_question = {
            "question": "테스트 문제",
            "options": ["1", "2", "3", "4", "5"],
            "correct_answer": 6,  # 1-5 범위를 벗어남
            "explanation": "테스트 해설",
            "difficulty": "medium",
            "subject": "수학",
            "unit": "일차함수"
        }

        result = self.generator.validate_question(invalid_question)
        assert result is False

    def test_validate_question_invalid_difficulty(self):
        """잘못된 난이도 검증 테스트"""
        invalid_question = {
            "question": "테스트 문제",
            "options": ["1", "2", "3", "4", "5"],
            "correct_answer": 1,
            "explanation": "테스트 해설",
            "difficulty": "invalid",  # 유효하지 않은 난이도
            "subject": "수학",
            "unit": "일차함수"
        }

        result = self.generator.validate_question(invalid_question)
        assert result is False

    def test_validate_question_empty_content(self):
        """빈 내용 검증 테스트"""
        invalid_question = {
            "question": "",  # 빈 문제
            "options": ["1", "2", "3", "4", "5"],
            "correct_answer": 1,
            "explanation": "테스트 해설",
            "difficulty": "medium",
            "subject": "수학",
            "unit": "일차함수"
        }

        result = self.generator.validate_question(invalid_question)
        assert result is False

    def test_get_question_statistics_empty(self):
        """빈 히스토리 통계 테스트"""
        stats = self.generator.get_question_statistics()

        assert stats['total_questions'] == 0
        assert stats['by_subject'] == {}
        assert stats['by_difficulty'] == {}
        assert stats['by_unit'] == {}

    def test_get_question_statistics_with_data(self):
        """데이터가 있는 히스토리 통계 테스트"""
        # 테스트 문제들을 히스토리에 추가
        test_questions = [
            {
                "subject": "수학",
                "difficulty": "easy",
                "unit": "일차함수",
                "generated_at": "2024-01-01T10:00:00"
            },
            {
                "subject": "수학",
                "difficulty": "medium",
                "unit": "일차함수",
                "generated_at": "2024-01-01T11:00:00"
            },
            {
                "subject": "과학",
                "difficulty": "easy",
                "unit": "물질의 상태",
                "generated_at": "2024-01-01T12:00:00"
            }
        ]

        self.generator.question_history = test_questions

        stats = self.generator.get_question_statistics()

        assert stats['total_questions'] == 3
        assert stats['by_subject']['수학'] == 2
        assert stats['by_subject']['과학'] == 1
        assert stats['by_difficulty']['easy'] == 2
        assert stats['by_difficulty']['medium'] == 1
        assert stats['by_unit']['일차함수'] == 2
        assert stats['by_unit']['물질의 상태'] == 1

    def test_create_question_prompt(self):
        """문제 생성 프롬프트 생성 테스트"""
        prompt = self.generator._create_question_prompt(
            subject="수학",
            unit="일차함수",
            difficulty="medium",
            context="일차함수는 y = ax + b 형태입니다."
        )

        assert "수학" in prompt
        assert "일차함수" in prompt
        assert "medium" in prompt
        assert "y = ax + b" in prompt
        assert "JSON" in prompt

    def test_validate_and_clean_question(self):
        """문제 검증 및 정리 테스트"""
        raw_response = {
            "question": "  테스트 문제  ",  # 앞뒤 공백
            "options": ["1", "2", "3", "4", "5"],
            "correct_answer": "2",  # 문자열 (정수로 변환되어야 함)
            "explanation": "  테스트 해설  ",  # 앞뒤 공백
            "difficulty": "medium",
            "subject": "수학",
            "unit": "일차함수"
        }

        cleaned = self.generator._validate_and_clean_question(
            raw_response, "수학", "일차함수", "medium"
        )

        assert cleaned["question"] == "테스트 문제"  # 공백 제거
        assert cleaned["explanation"] == "테스트 해설"  # 공백 제거
        assert cleaned["correct_answer"] == 2  # 정수로 변환
        assert "generated_at" in cleaned
        assert "id" in cleaned

    def test_generate_varied_query(self):
        """다양한 쿼리 생성 테스트"""
        query1 = self.generator._generate_varied_query("수학", "일차함수", 0)
        query2 = self.generator._generate_varied_query("수학", "일차함수", 1)
        query3 = self.generator._generate_varied_query("수학", "일차함수", 8)  # 순환 테스트

        assert "수학" in query1 and "일차함수" in query1
        assert "수학" in query2 and "일차함수" in query2
        assert query1 != query2  # 다른 쿼리가 생성되어야 함

    def test_add_to_history(self):
        """히스토리 추가 테스트"""
        question = {
            "question": "테스트 문제",
            "subject": "수학",
            "unit": "일차함수"
        }

        self.generator._add_to_history(question)

        assert len(self.generator.question_history) == 1
        assert self.generator.question_history[0] == question

    def test_history_size_limit(self):
        """히스토리 크기 제한 테스트"""
        # 1000개보다 많은 문제 추가
        for i in range(1001):
            question = {"id": i, "question": f"문제 {i}"}
            self.generator._add_to_history(question)

        # 1000개로 제한되어야 함
        assert len(self.generator.question_history) == 1000
        # 가장 오래된 것이 제거되고 최근 1000개만 남아야 함
        assert self.generator.question_history[0]["id"] == 1

    def test_generate_question_with_hint(self):
        """hint 필드가 포함된 문제 생성 테스트"""
        # Mock 설정
        self.mock_retriever.retrieve_context.return_value = [
            "일차함수는 y = ax + b 형태입니다.",
            "기울기 a는 직선의 기울어진 정도를 나타냅니다."
        ]

        mock_response = {
            "question": "일차함수 y = 3x - 2에서 y절편은 무엇인가?",
            "options": ["3", "-2", "2", "-3", "0"],
            "correct_answer": 2,
            "explanation": "일차함수 y = ax + b에서 b가 y절편이므로, y = 3x - 2에서 y절편은 -2입니다.",
            "hint": "y절편은 직선이 y축과 만나는 점의 y좌표입니다.",
            "difficulty": "medium",
            "subject": "수학",
            "unit": "일차함수"
        }

        self.mock_llm_client.generate_structured_response.return_value = mock_response

        # 문제 생성 실행
        result = self.generator.generate_question(
            subject="수학",
            unit="일차함수",
            difficulty="medium"
        )

        # hint 필드 검증
        assert "hint" in result
        assert result["hint"] == "y절편은 직선이 y축과 만나는 점의 y좌표입니다."
        assert result["hint"].strip() != ""

    def test_generate_question_without_hint(self):
        """hint가 없는 문제 생성 테스트"""
        # Mock 설정
        self.mock_retriever.retrieve_context.return_value = ["테스트 컨텍스트"]

        mock_response = {
            "question": "테스트 문제",
            "options": ["1", "2", "3", "4", "5"],
            "correct_answer": 1,
            "explanation": "테스트 해설",
            "difficulty": "easy",
            "subject": "수학",
            "unit": "일차함수"
            # hint 필드 없음
        }

        self.mock_llm_client.generate_structured_response.return_value = mock_response

        # 문제 생성 실행
        result = self.generator.generate_question(
            subject="수학",
            unit="일차함수",
            difficulty="easy"
        )

        # hint 필드가 빈 문자열로 설정되어야 함
        assert "hint" in result
        assert result["hint"] == ""

    def test_validate_question_with_hint(self):
        """hint가 포함된 문제 검증 테스트"""
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

        assert self.generator.validate_question(valid_question) == True

    def test_validate_question_without_hint(self):
        """hint가 없는 문제 검증 테스트 (hint는 선택사항)"""
        valid_question = {
            "question": "테스트 문제",
            "options": ["A", "B", "C", "D", "E"],
            "correct_answer": 1,
            "explanation": "테스트 해설",
            "difficulty": "medium",
            "subject": "수학",
            "unit": "일차함수"
            # hint 필드 없음 - 이것도 유효해야 함
        }

        assert self.generator.validate_question(valid_question) == True