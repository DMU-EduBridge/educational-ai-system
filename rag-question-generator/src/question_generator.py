"""
Question Generator - 5지선다 문제 생성기
RAG 기반으로 다양한 유형의 문제를 생성합니다.
"""

import json
import random
from typing import List, Dict, Any
import openai
from rag_processor import RAGProcessor


class QuestionGenerator:
    """5지선다 문제 생성기"""

    def __init__(self, rag_processor: RAGProcessor, llm_model: str = "gpt-3.5-turbo"):
        """
        Args:
            rag_processor: RAG 처리기
            llm_model: 사용할 LLM 모델
        """
        self.rag_processor = rag_processor
        self.llm_model = llm_model
        self.client = openai.OpenAI()

        # 문제 유형별 프롬프트 템플릿
        self.QUESTION_GENERATION_PROMPTS = {
            "concept": """
다음 텍스트를 바탕으로 핵심 개념을 묻는 5지선다 문제를 생성하세요.

텍스트:
{context}

요구사항:
1. 텍스트에 직접 언급된 개념을 묻는 문제
2. 명확하고 이해하기 쉬운 문제
3. 5개의 선택지 (정답 1개, 오답 4개)
4. 오답은 그럴듯하지만 명확히 틀린 것
5. 상세한 해설 포함

JSON 형태로만 출력:
{{
    "question": "문제 텍스트",
    "options": ["선택지1", "선택지2", "선택지3", "선택지4", "선택지5"],
    "correct_answer": 정답_번호(1-5),
    "explanation": "정답 해설",
    "difficulty": "easy|medium|hard",
    "type": "concept"
}}
""",

            "application": """
다음 텍스트를 바탕으로 응용/적용 문제를 생성하세요.

텍스트:
{context}

요구사항:
1. 텍스트의 개념을 다른 상황에 적용하는 문제
2. 단순 암기가 아닌 이해를 요구하는 문제
3. 5개의 선택지 (정답 1개, 오답 4개)
4. 실생활 연결 가능한 문제
5. 상세한 해설 포함

JSON 형태로만 출력:
{{
    "question": "문제 텍스트",
    "options": ["선택지1", "선택지2", "선택지3", "선택지4", "선택지5"],
    "correct_answer": 정답_번호(1-5),
    "explanation": "정답 해설",
    "difficulty": "easy|medium|hard",
    "type": "application"
}}
""",

            "inference": """
다음 텍스트를 바탕으로 추론 문제를 생성하세요.

텍스트:
{context}

요구사항:
1. 텍스트 내용을 바탕으로 논리적 추론을 요구하는 문제
2. 직접적으로 언급되지 않았지만 유추 가능한 내용
3. 5개의 선택지 (정답 1개, 오답 4개)
4. 비판적 사고를 요구하는 문제
5. 상세한 해설 포함

JSON 형태로만 출력:
{{
    "question": "문제 텍스트",
    "options": ["선택지1", "선택지2", "선택지3", "선택지4", "선택지5"],
    "correct_answer": 정답_번호(1-5),
    "explanation": "정답 해설",
    "difficulty": "easy|medium|hard",
    "type": "inference"
}}
"""
        }

        # 난이도별 분배 설정
        self.difficulty_distributions = {
            "balanced": {"easy": 3, "medium": 4, "hard": 3},
            "easy": {"easy": 7, "medium": 2, "hard": 1},
            "medium": {"easy": 2, "medium": 6, "hard": 2},
            "hard": {"easy": 1, "medium": 2, "hard": 7}
        }

    def generate_questions(self, num_questions: int = 10, difficulty_mix: str = "balanced") -> List[Dict[str, Any]]:
        """
        RAG 기반 5지선다 문제 10개 생성

        Args:
            num_questions: 생성할 문제 수 (기본 10개)
            difficulty_mix: 난이도 구성 (easy/medium/hard/balanced)

        Returns:
            List[dict]: 생성된 문제들
        """
        if not self.rag_processor.collection:
            raise ValueError("문서가 처리되지 않았습니다. RAGProcessor.process_document()를 먼저 실행하세요.")

        print(f"📝 {num_questions}개의 문제 생성을 시작합니다...")

        # 난이도 분배 계획 수립
        difficulty_plan = self._create_difficulty_plan(num_questions, difficulty_mix)
        print(f"🎯 난이도 분배: {difficulty_plan}")

        # 문제 유형 분배 계획
        type_plan = self._create_type_plan(num_questions)
        print(f"📋 문제 유형 분배: {type_plan}")

        questions = []
        used_contexts = set()

        for i in range(num_questions):
            try:
                # 현재 문제의 난이도와 유형 결정
                current_difficulty = self._get_next_difficulty(difficulty_plan, i)
                current_type = self._get_next_type(type_plan, i)

                print(f"🔄 문제 {i+1}/{num_questions} 생성 중... ({current_type}, {current_difficulty})")

                # 컨텍스트 검색
                context = self._get_unique_context(used_contexts)
                if not context:
                    print(f"⚠️ 고유 컨텍스트를 찾을 수 없습니다. 문제 {i+1} 건너뜀")
                    continue

                # 문제 생성
                question_data = self._generate_single_question(context, current_type, current_difficulty)

                if question_data and self._validate_question(question_data):
                    question_data["id"] = i + 1
                    questions.append(question_data)
                    used_contexts.add(context[:100])  # 앞 100자로 중복 체크
                    print(f"✅ 문제 {i+1} 생성 완료")
                else:
                    print(f"❌ 문제 {i+1} 생성 실패")

            except Exception as e:
                print(f"❌ 문제 {i+1} 생성 중 오류: {str(e)}")
                continue

        print(f"🎉 총 {len(questions)}개 문제 생성 완료!")
        return questions

    def _create_difficulty_plan(self, num_questions: int, difficulty_mix: str) -> List[str]:
        """난이도 분배 계획 생성"""
        if difficulty_mix in self.difficulty_distributions:
            distribution = self.difficulty_distributions[difficulty_mix]
        else:
            distribution = self.difficulty_distributions["balanced"]

        plan = []
        for difficulty, count in distribution.items():
            # 비율에 맞춰 조정
            actual_count = int(count * num_questions / 10)
            plan.extend([difficulty] * actual_count)

        # 남은 문제는 medium으로 채움
        while len(plan) < num_questions:
            plan.append("medium")

        # 셔플해서 순서 무작위화
        random.shuffle(plan)
        return plan

    def _create_type_plan(self, num_questions: int) -> List[str]:
        """문제 유형 분배 계획 생성"""
        types = ["concept", "application", "inference"]
        plan = []

        # 균등 분배
        base_count = num_questions // 3
        remainder = num_questions % 3

        for i, question_type in enumerate(types):
            count = base_count + (1 if i < remainder else 0)
            plan.extend([question_type] * count)

        random.shuffle(plan)
        return plan

    def _get_next_difficulty(self, plan: List[str], index: int) -> str:
        """다음 문제의 난이도 반환"""
        return plan[index] if index < len(plan) else "medium"

    def _get_next_type(self, plan: List[str], index: int) -> str:
        """다음 문제의 유형 반환"""
        return plan[index] if index < len(plan) else "concept"

    def _get_unique_context(self, used_contexts: set) -> str:
        """중복되지 않는 컨텍스트 선택"""
        all_chunks = self.rag_processor.get_all_chunks()

        for chunk in all_chunks:
            chunk_start = chunk[:100]
            if chunk_start not in used_contexts:
                return chunk

        # 모든 청크가 사용된 경우 랜덤 선택
        return random.choice(all_chunks) if all_chunks else ""

    def _generate_single_question(self, context: str, question_type: str, difficulty: str) -> Dict[str, Any]:
        """단일 문제 생성"""
        try:
            prompt = self._create_question_prompt(context, question_type)

            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "당신은 교육 전문가입니다. 주어진 텍스트를 바탕으로 고품질의 5지선다 문제를 생성합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            content = response.choices[0].message.content.strip()

            # JSON 파싱
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]

            question_data = json.loads(content)

            # 난이도 강제 설정 (LLM이 잘못 설정할 수 있음)
            question_data["difficulty"] = difficulty

            return question_data

        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 오류: {e}")
            return None
        except Exception as e:
            print(f"❌ 문제 생성 오류: {e}")
            return None

    def _create_question_prompt(self, context: str, question_type: str) -> str:
        """
        문제 생성용 프롬프트 생성

        Args:
            context: RAG에서 검색된 컨텍스트
            question_type: 문제 유형

        Returns:
            str: 완성된 프롬프트
        """
        if question_type not in self.QUESTION_GENERATION_PROMPTS:
            question_type = "concept"

        return self.QUESTION_GENERATION_PROMPTS[question_type].format(context=context)

    def _validate_question(self, question_data: Dict[str, Any]) -> bool:
        """
        생성된 문제 검증

        Args:
            question_data: 문제 데이터

        Returns:
            bool: 유효성 검사 결과
        """
        required_fields = ["question", "options", "correct_answer", "explanation", "difficulty", "type"]

        # 필수 필드 확인
        for field in required_fields:
            if field not in question_data:
                print(f"❌ 필수 필드 누락: {field}")
                return False

        # 선택지 개수 확인
        if not isinstance(question_data["options"], list) or len(question_data["options"]) != 5:
            print(f"❌ 선택지는 정확히 5개여야 합니다.")
            return False

        # 정답 번호 확인
        correct_answer = question_data["correct_answer"]
        if not isinstance(correct_answer, int) or correct_answer < 1 or correct_answer > 5:
            print(f"❌ 정답 번호는 1-5 사이여야 합니다.")
            return False

        # 문제 텍스트 길이 확인
        if len(question_data["question"]) < 10:
            print(f"❌ 문제 텍스트가 너무 짧습니다.")
            return False

        # 선택지 중복 확인
        options = question_data["options"]
        if len(set(options)) != len(options):
            print(f"❌ 선택지에 중복이 있습니다.")
            return False

        return True