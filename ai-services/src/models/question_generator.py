from typing import Dict, List, Optional, Any
import json
import logging
import re
from datetime import datetime

from .llm_client import LLMClient
from ..rag.retriever import RAGRetriever


class QuestionGenerator:
    """5지선다 문제 생성기"""

    def __init__(self,
                 llm_client: LLMClient,
                 retriever: RAGRetriever):
        """
        QuestionGenerator 초기화

        Args:
            llm_client: LLMClient 인스턴스
            retriever: RAGRetriever 인스턴스
        """
        self.llm_client = llm_client
        self.retriever = retriever
        self.logger = logging.getLogger(__name__)

        # 생성된 문제 히스토리
        self.question_history = []

    def generate_question(self,
                         subject: str,
                         unit: str,
                         difficulty: str = "medium",
                         custom_query: Optional[str] = None) -> Dict[str, Any]:
        """
        5지선다 문제 생성

        Args:
            subject: 과목명
            unit: 단원명
            difficulty: 난이도 (easy, medium, hard)
            custom_query: 커스텀 검색 쿼리

        Returns:
            Dict[str, Any]: 생성된 문제 데이터
        """
        try:
            # 검색 쿼리 준비
            if custom_query:
                search_query = custom_query
            else:
                search_query = f"{subject} {unit} 개념"

            # 관련 컨텍스트 검색
            retrieved_docs = self.retriever.retrieve_documents(
                query=search_query,
                subject=subject,
                unit=unit,
                k=3
            )

            if not retrieved_docs:
                raise ValueError(f"No context found for {subject} - {unit}")

            # 컨텍스트 포맷팅
            context = self.retriever.format_context(retrieved_docs)

            # 프롬프트 생성
            prompt = self._create_question_prompt(
                subject=subject,
                unit=unit,
                difficulty=difficulty,
                context=context
            )

            # LLM으로 문제 생성
            response = self.llm_client.generate_structured_response(
                prompt=prompt,
                response_format="json",
                max_tokens=1500
            )

            # 응답 검증 및 후처리
            validated_question = self._validate_and_clean_question(response, subject, unit, difficulty)

            # 히스토리에 추가
            self._add_to_history(validated_question)

            self.logger.info(f"Generated question for {subject} - {unit} ({difficulty})")
            return validated_question

        except Exception as e:
            self.logger.error(f"Error generating question: {str(e)}")
            raise

    def generate_batch_questions(self,
                               subject: str,
                               unit: str,
                               count: int = 5,
                               difficulty: str = "medium") -> List[Dict[str, Any]]:
        """
        배치 문제 생성

        Args:
            subject: 과목명
            unit: 단원명
            count: 생성할 문제 수
            difficulty: 난이도

        Returns:
            List[Dict[str, Any]]: 생성된 문제 리스트
        """
        try:
            questions = []
            failed_attempts = 0
            max_failures = count * 2  # 실패 허용 횟수

            for i in range(count):
                try:
                    # 각 문제마다 다른 검색 키워드 사용
                    custom_query = self._generate_varied_query(subject, unit, i)

                    question = self.generate_question(
                        subject=subject,
                        unit=unit,
                        difficulty=difficulty,
                        custom_query=custom_query
                    )

                    questions.append(question)
                    self.logger.info(f"Generated question {i+1}/{count}")

                except Exception as e:
                    failed_attempts += 1
                    self.logger.warning(f"Failed to generate question {i+1}: {str(e)}")

                    if failed_attempts >= max_failures:
                        self.logger.error("Too many failures, stopping batch generation")
                        break

            self.logger.info(f"Batch generation completed: {len(questions)}/{count} questions generated")
            return questions

        except Exception as e:
            self.logger.error(f"Error in batch question generation: {str(e)}")
            raise

    def validate_question(self, question_data: Dict[str, Any]) -> bool:
        """
        문제 데이터 검증

        Args:
            question_data: 문제 데이터

        Returns:
            bool: 유효성 여부
        """
        try:
            required_fields = ['question', 'options', 'correct_answer', 'explanation', 'difficulty', 'subject', 'unit']
            optional_fields = ['hint']  # hint는 선택사항

            # 필수 필드 확인
            for field in required_fields:
                if field not in question_data:
                    self.logger.error(f"Missing required field: {field}")
                    return False

            # 문제 텍스트 확인
            if not question_data['question'].strip():
                self.logger.error("Question text is empty")
                return False

            # 선택지 확인
            options = question_data['options']
            if not isinstance(options, list) or len(options) != 5:
                self.logger.error("Options must be a list with exactly 5 items")
                return False

            for i, option in enumerate(options, 1):
                if not isinstance(option, str) or not option.strip():
                    self.logger.error(f"Option {i} is empty or not a string")
                    return False

            # 정답 확인
            correct_answer = question_data['correct_answer']
            if not isinstance(correct_answer, int) or correct_answer < 1 or correct_answer > 5:
                self.logger.error("Correct answer must be an integer between 1 and 5")
                return False

            # 해설 확인
            if not question_data['explanation'].strip():
                self.logger.error("Explanation is empty")
                return False

            # 난이도 확인
            if question_data['difficulty'] not in ['easy', 'medium', 'hard']:
                self.logger.error("Difficulty must be 'easy', 'medium', or 'hard'")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating question: {str(e)}")
            return False

    def get_question_statistics(self) -> Dict[str, Any]:
        """
        문제 생성 통계 반환

        Returns:
            Dict[str, Any]: 통계 정보
        """
        if not self.question_history:
            return {
                'total_questions': 0,
                'by_subject': {},
                'by_difficulty': {},
                'by_unit': {},
                'generation_times': []
            }

        stats = {
            'total_questions': len(self.question_history),
            'by_subject': {},
            'by_difficulty': {},
            'by_unit': {},
            'generation_times': []
        }

        for question in self.question_history:
            # 과목별 통계
            subject = question.get('subject', 'Unknown')
            stats['by_subject'][subject] = stats['by_subject'].get(subject, 0) + 1

            # 난이도별 통계
            difficulty = question.get('difficulty', 'Unknown')
            stats['by_difficulty'][difficulty] = stats['by_difficulty'].get(difficulty, 0) + 1

            # 단원별 통계
            unit = question.get('unit', 'Unknown')
            stats['by_unit'][unit] = stats['by_unit'].get(unit, 0) + 1

            # 생성 시간
            if 'generated_at' in question:
                stats['generation_times'].append(question['generated_at'])

        return stats

    def _create_question_prompt(self,
                              subject: str,
                              unit: str,
                              difficulty: str,
                              context: str) -> str:
        """
        문제 생성용 프롬프트 생성

        Args:
            subject: 과목명
            unit: 단원명
            difficulty: 난이도
            context: 교과서 컨텍스트

        Returns:
            str: 생성된 프롬프트
        """
        # 난이도별 가이드라인
        difficulty_guidelines = {
            'easy': '기본 개념 이해 확인, 단순 암기, 용어 정의',
            'medium': '개념 적용 및 계산, 예제 문제 응용',
            'hard': '복합적 사고 및 응용, 심화 분석, 문제 해결'
        }

        difficulty_guide = difficulty_guidelines.get(difficulty, difficulty_guidelines['medium'])

        prompt = f"""당신은 중학교 {subject} 과목의 전문 교사입니다.
다음 교과서 내용을 바탕으로 {difficulty} 난이도의 5지선다 문제를 1개 생성해주세요.

교과서 내용:
{context}

문제 생성 규칙:
1. 교과서 내용에 직접 관련된 문제
2. 중학교 1학년 수준에 맞는 명확한 문제
3. 5개의 선택지 (정답 1개, 매력적인 오답 4개)
4. 상세하고 교육적인 해설
5. 문제 해결에 도움이 되는 힌트 (선택사항)
6. 한국어로 작성

난이도 기준 ({difficulty}):
{difficulty_guide}

선택지 작성 가이드:
- 정답: 교과서 내용과 완전히 일치하는 올바른 답
- 오답: 일부분만 맞거나, 일반적인 오개념, 유사하지만 틀린 내용
- 모든 선택지는 문법적으로 자연스럽고 길이가 비슷해야 함

힌트 작성 가이드:
- 직접적인 정답을 제시하지 않으면서 문제 해결 방향 제시
- 관련 개념이나 공식에 대한 간접적인 언급
- 문제를 푸는 데 도움이 되는 사고 과정 유도
- 너무 명확하지 않게, 학습자의 사고를 자극하는 수준

출력 형식 (JSON만 출력):
{{
    "question": "문제 텍스트",
    "options": ["1번 선택지", "2번 선택지", "3번 선택지", "4번 선택지", "5번 선택지"],
    "correct_answer": 정답_번호(1-5),
    "explanation": "정답 해설 및 풀이 과정",
    "hint": "문제 해결을 위한 힌트 (선택사항)",
    "difficulty": "{difficulty}",
    "subject": "{subject}",
    "unit": "{unit}"
}}"""

        return prompt

    def _validate_and_clean_question(self,
                                   response: Dict[str, Any],
                                   subject: str,
                                   unit: str,
                                   difficulty: str) -> Dict[str, Any]:
        """
        생성된 문제 검증 및 정리

        Args:
            response: LLM 응답
            subject: 과목명
            unit: 단원명
            difficulty: 난이도

        Returns:
            Dict[str, Any]: 검증된 문제 데이터
        """
        # 기본값 설정
        cleaned_question = {
            'question': response.get('question', '').strip(),
            'options': response.get('options', []),
            'correct_answer': response.get('correct_answer', 1),
            'explanation': response.get('explanation', '').strip(),
            'hint': response.get('hint', '').strip(),  # hint 필드 추가 (선택사항)
            'difficulty': difficulty,
            'subject': subject,
            'unit': unit,
            'generated_at': datetime.now().isoformat(),
            'id': f"{subject}_{unit}_{difficulty}_{len(self.question_history) + 1}"
        }

        # 옵션 정리
        if isinstance(cleaned_question['options'], list):
            cleaned_question['options'] = [str(opt).strip() for opt in cleaned_question['options']]

        # 정답 번호 검증
        try:
            cleaned_question['correct_answer'] = int(cleaned_question['correct_answer'])
        except (ValueError, TypeError):
            cleaned_question['correct_answer'] = 1

        # 검증 수행
        if not self.validate_question(cleaned_question):
            raise ValueError("Generated question failed validation")

        return cleaned_question

    def _generate_varied_query(self, subject: str, unit: str, index: int) -> str:
        """
        다양한 검색 쿼리 생성

        Args:
            subject: 과목명
            unit: 단원명
            index: 문제 인덱스

        Returns:
            str: 검색 쿼리
        """
        query_variations = [
            f"{subject} {unit} 개념",
            f"{subject} {unit} 예제",
            f"{subject} {unit} 응용",
            f"{subject} {unit} 문제",
            f"{subject} {unit} 정의",
            f"{subject} {unit} 계산",
            f"{subject} {unit} 공식",
            f"{subject} {unit} 원리"
        ]

        return query_variations[index % len(query_variations)]

    def _add_to_history(self, question: Dict[str, Any]):
        """
        문제를 히스토리에 추가

        Args:
            question: 문제 데이터
        """
        self.question_history.append(question)

        # 히스토리 크기 제한 (최대 1000개)
        if len(self.question_history) > 1000:
            self.question_history = self.question_history[-1000:]