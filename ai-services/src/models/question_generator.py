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
        문제 데이터 검증 (모든 필드 필수)

        Args:
            question_data: 문제 데이터

        Returns:
            bool: 유효성 여부
        """
        try:
            # 모든 필드를 필수로 간주
            required_fields = [
                'title', 'description', 'content', 'type', 'difficulty', 'subject',
                'gradeLevel', 'unit', 'options', 'correctAnswer', 'explanation',
                'hints', 'tags', 'points', 'timeLimit', 'isAIGenerated'
            ]

            # 필수 필드 존재 여부 확인
            for field in required_fields:
                if field not in question_data or question_data[field] is None:
                    self.logger.error(f"Missing required field: {field}")
                    return False

            # 내용 확인 (비어 있으면 안 됨)
            if not all(question_data[f].strip() for f in ['title', 'content', 'explanation']):
                self.logger.error("Title, content, or explanation is empty")
                return False
            
            # Description은 비어있을 수 있으나, None이어서는 안됨
            if question_data['description'] is None:
                self.logger.error("Description field is missing")
                return False

            # 선택지 확인 (5지선다, 비어 있으면 안 됨)
            options = question_data['options']
            if not isinstance(options, list) or len(options) != 5:
                self.logger.error("Options must be a list with exactly 5 items")
                return False
            if not all(isinstance(opt, str) and opt.strip() for opt in options):
                self.logger.error("All options must be non-empty strings")
                return False

            # 정답 확인 (문자열 형태의 숫자 1-5)
            correct_answer = question_data['correctAnswer']
            if not isinstance(correct_answer, str) or not correct_answer.isdigit() or not 1 <= int(correct_answer) <= 5:
                self.logger.error("Correct answer must be a string representing an integer between 1 and 5")
                return False

            # JSON 형태의 리스트 필드 확인 (비어 있으면 안 됨)
            for field in ['hints', 'tags']:
                if not isinstance(question_data[field], list) or not question_data[field]:
                    self.logger.error(f"Field '{field}' must be a non-empty list")
                    return False

            # 숫자형 필드 확인
            if not all(isinstance(question_data[f], int) for f in ['points', 'timeLimit']):
                self.logger.error("Fields 'points' and 'timeLimit' must be integers")
                return False

            # 기타 필드 확인
            if question_data['difficulty'] not in ['easy', 'medium', 'hard']:
                self.logger.error("Difficulty must be 'easy', 'medium', or 'hard'")
                return False
            if question_data['type'] not in ['multiple_choice', 'short_answer', 'essay']:
                self.logger.error("Invalid problem type")
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
            if 'createdAt' in question:
                stats['generation_times'].append(question['createdAt'])

        return stats

    def _create_question_prompt(self,
                              subject: str,
                              unit: str,
                              difficulty: str,
                              context: str) -> str:
        """
        문제 생성용 프롬프트 생성 (모든 필드 필수)

        Args:
            subject: 과목명
            unit: 단원명
            difficulty: 난이도
            context: 교과서 컨텍스트

        Returns:
            str: 생성된 프롬프트
        """
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
1. 교과서 내용에 직접 관련된 문제.
2. 중학교 1학년 수준에 맞는 명확한 문제.
3. 5개의 선택지 (정답 1개, 매력적인 오답 4개).
4. 상세하고 교육적인 해설.
5. 문제 해결에 도움이 되는 힌트 목록 (최소 1개 이상).
6. 문제의 핵심 내용을 담은 간결한 제목.
7. 문제에 대한 부가적인 설명 (description).
8. 관련 개념을 나타내는 태그 목록 (최소 1개 이상).
9. 모든 내용은 한국어로 작성.

난이도 기준 ({difficulty}):
{difficulty_guide}

출력 형식 (JSON만 출력, 다른 설명 없이 JSON 객체만 반환):
{{
    "title": "문제의 간결한 제목",
    "description": "문제에 대한 부가적인 설명입니다.",
    "content": "여기에 문제의 본문을 작성합니다.",
    "options": ["1번 선택지", "2번 선택지", "3번 선택지", "4번 선택지", "5번 선택지"],
    "correct_answer": 정답_번호(1-5 사이의 숫자),
    "explanation": "정답에 대한 상세하고 친절한 해설입니다.",
    "hints": ["문제 해결에 도움이 되는 첫 번째 힌트"],
    "tags": ["관련_태그_1"]
}}
"""
        return prompt

    def _validate_and_clean_question(self,
                                   response: Dict[str, Any],
                                   subject: str,
                                   unit: str,
                                   difficulty: str) -> Dict[str, Any]:
        """
        생성된 문제 검증 및 정리 (모든 필드 필수)

        Args:
            response: LLM 응답
            subject: 과목명
            unit: 단원명
            difficulty: 난이도

        Returns:
            Dict[str, Any]: 검증 및 변환된 문제 데이터
        """
        now = datetime.now().isoformat()
        ai_generation_id = f"{subject}_{unit}_{difficulty}_{len(self.question_history) + 1}"

        # LLM 응답에서 데이터 추출 및 기본값 설정
        title = response.get('title', 'Untitled').strip()
        description = response.get('description', '').strip()
        content = response.get('content', '').strip()
        options = response.get('options', [])
        correct_answer_num = response.get('correct_answer', 1)
        explanation = response.get('explanation', '').strip()
        hints = response.get('hints', [])
        tags = response.get('tags', [])

        # 데이터 변환 및 추가 필드 설정
        problem_data = {
            'id': None,  # DB에서 자동 생성
            'title': title if title else content[:50], # 제목 없으면 내용에서 일부 추출
            'description': description,
            'content': content,
            'type': 'multiple_choice', # 5지선다 유형
            'difficulty': difficulty,
            'subject': subject,
            'gradeLevel': 'Middle-1', # 중학교 1학년으로 가정
            'unit': unit,
            'options': [str(opt).strip() for opt in options] if isinstance(options, list) else [],
            'correctAnswer': str(correct_answer_num), # DB 스키마에 맞춰 문자열로 변환
            'explanation': explanation,
            'hints': [str(h).strip() for h in hints] if isinstance(hints, list) else [],
            'tags': [str(t).strip() for t in tags] if isinstance(tags, list) else [],
            'points': 10, # 기본 점수
            'timeLimit': 60, # 기본 제한 시간 (초)
            'isActive': True,
            'isAIGenerated': True,
            'aiGenerationId': ai_generation_id,
            'qualityScore': None,
            'reviewStatus': 'pending', # 검토 대기 상태
            'reviewedAt': None,
            'generationPrompt': None, # 필요시 프롬프트 저장
            'contextChunkIds': None, # 필요시 컨텍스트 ID 저장
            'modelName': self.llm_client.model_name,
            'createdAt': now,
            'updatedAt': now,
            'deletedAt': None
        }
        
        # 정답 번호 검증 (1-5 사이의 숫자인지)
        try:
            if not (1 <= int(problem_data['correctAnswer']) <= 5):
                problem_data['correctAnswer'] = '1'
        except (ValueError, TypeError):
            problem_data['correctAnswer'] = '1'

        # 최종 데이터 유효성 검사
        if not self.validate_question(problem_data):
            raise ValueError("Generated question failed validation")

        return problem_data

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