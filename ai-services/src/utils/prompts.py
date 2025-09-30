"""
교육용 AI 시스템을 위한 프롬프트 템플릿 모듈
"""

from typing import Dict, Any, Optional
import json


class PromptTemplate:
    """프롬프트 템플릿 기본 클래스"""

    def __init__(self, template: str, required_variables: list = None):
        self.template = template
        self.required_variables = required_variables or []

    def format(self, **kwargs) -> str:
        """템플릿에 변수를 적용하여 프롬프트 생성"""
        # 필수 변수 확인
        missing_vars = [var for var in self.required_variables if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")

        return self.template.format(**kwargs)

    def validate_variables(self, **kwargs) -> bool:
        """변수 유효성 검사"""
        for var in self.required_variables:
            if var not in kwargs or not kwargs[var]:
                return False
        return True


# 5지선다 문제 생성용 메인 프롬프트
QUESTION_GENERATION_PROMPT = PromptTemplate(
    template="""당신은 중학교 {subject} 과목의 전문 교사입니다.
다음 교과서 내용을 바탕으로 {difficulty} 난이도의 5지선다 문제를 1개 생성해주세요.

교과서 내용:
{context}

문제 생성 규칙:
1. 교과서 내용에 직접 관련된 문제
2. 중학교 1학년 수준에 맞는 명확한 문제
3. 5개의 선택지 (정답 1개, 매력적인 오답 4개)
4. 상세하고 교육적인 해설
5. 한국어로 작성

난이도 기준:
- easy: 기본 개념 이해 확인
- medium: 개념 적용 및 계산
- hard: 복합적 사고 및 응용

출력 형식 (JSON만 출력):
{{
    "question": "문제 텍스트",
    "options": ["1번", "2번", "3번", "4번", "5번"],
    "correct_answer": 정답_번호(1-5),
    "explanation": "정답 해설 및 풀이 과정",
    "difficulty": "{difficulty}",
    "subject": "{subject}",
    "unit": "{unit}"
}}""",
    required_variables=['subject', 'difficulty', 'context', 'unit']
)

# 수학 과목 전용 프롬프트
MATH_QUESTION_PROMPT = PromptTemplate(
    template="""당신은 중학교 수학 전문 교사입니다.
다음 교과서 내용을 바탕으로 {difficulty} 난이도의 수학 문제를 생성해주세요.

교과서 내용:
{context}

수학 문제 생성 특별 규칙:
1. 수식과 계산이 포함된 문제
2. 단계별 풀이 과정 제시
3. 일반적인 실수 유형을 오답에 반영
4. 공식이나 정리 활용 문제

난이도별 가이드:
- easy: 기본 공식 적용, 단순 계산
- medium: 복합 계산, 응용 문제
- hard: 증명, 심화 응용, 문제 해결

출력 형식 (JSON만 출력):
{{
    "question": "문제 텍스트 (수식은 일반 텍스트로)",
    "options": ["1번", "2번", "3번", "4번", "5번"],
    "correct_answer": 정답_번호(1-5),
    "explanation": "단계별 풀이 과정",
    "difficulty": "{difficulty}",
    "subject": "수학",
    "unit": "{unit}",
    "math_concept": "관련 수학 개념"
}}""",
    required_variables=['difficulty', 'context', 'unit']
)

# 과학 과목 전용 프롬프트
SCIENCE_QUESTION_PROMPT = PromptTemplate(
    template="""당신은 중학교 과학 전문 교사입니다.
다음 교과서 내용을 바탕으로 {difficulty} 난이도의 과학 문제를 생성해주세요.

교과서 내용:
{context}

과학 문제 생성 특별 규칙:
1. 과학적 현상과 원리 이해 확인
2. 실생활 연관 사례 활용
3. 실험과 관찰 결과 해석
4. 과학적 사고력 평가

난이도별 가이드:
- easy: 기본 개념, 용어 정의
- medium: 현상 설명, 원리 적용
- hard: 실험 설계, 결과 분석

출력 형식 (JSON만 출력):
{{
    "question": "문제 텍스트",
    "options": ["1번", "2번", "3번", "4번", "5번"],
    "correct_answer": 정답_번호(1-5),
    "explanation": "과학적 원리 설명",
    "difficulty": "{difficulty}",
    "subject": "과학",
    "unit": "{unit}",
    "science_field": "관련 과학 분야"
}}""",
    required_variables=['difficulty', 'context', 'unit']
)

# 컨텍스트 요약용 프롬프트
CONTEXT_SUMMARY_PROMPT = PromptTemplate(
    template="""다음 교과서 내용을 {length} 길이로 요약해주세요.

원본 내용:
{content}

요약 조건:
1. 핵심 개념과 중요한 정보 위주로 요약
2. 학습 목표와 연관된 내용 강조
3. 명확하고 이해하기 쉬운 문장으로 작성
4. {subject} 과목의 {unit} 단원 내용임을 고려

요약 길이: {length}
- short: 2-3 문장
- medium: 4-6 문장
- long: 7-10 문장

요약 결과:""",
    required_variables=['content', 'length', 'subject', 'unit']
)

# 문제 검증용 프롬프트
QUESTION_VALIDATION_PROMPT = PromptTemplate(
    template="""다음 5지선다 문제를 교육 전문가 관점에서 검증해주세요.

문제:
{question_json}

검증 항목:
1. 문제 명확성: 문제가 명확하고 이해하기 쉬운가?
2. 선택지 품질: 정답과 오답이 적절한가?
3. 난이도 적합성: 제시된 난이도에 맞는가?
4. 교육적 가치: 학습 목표 달성에 도움이 되는가?
5. 언어 수준: 중학교 1학년 수준에 적합한가?

JSON 형식으로 응답:
{{
    "is_valid": true/false,
    "score": 점수(1-10),
    "feedback": {{
        "clarity": "명확성 평가",
        "options_quality": "선택지 품질 평가",
        "difficulty": "난이도 평가",
        "educational_value": "교육적 가치 평가",
        "language_level": "언어 수준 평가"
    }},
    "suggestions": ["개선 제안 1", "개선 제안 2"],
    "overall_comment": "전체 평가"
}}""",
    required_variables=['question_json']
)

# 키워드 추출용 프롬프트
KEYWORD_EXTRACTION_PROMPT = PromptTemplate(
    template="""다음 {subject} 교과서 내용에서 핵심 키워드를 추출해주세요.

교과서 내용:
{content}

추출 조건:
1. 학습 목표와 직접 관련된 키워드
2. 문제 출제 가능한 중요 개념
3. 과목 특성을 반영한 전문 용어
4. 5-10개의 키워드 선정

JSON 형식으로 응답:
{{
    "primary_keywords": ["핵심 키워드 1", "핵심 키워드 2"],
    "secondary_keywords": ["보조 키워드 1", "보조 키워드 2"],
    "concepts": ["개념 1", "개념 2"],
    "terms": ["전문 용어 1", "전문 용어 2"]
}}""",
    required_variables=['subject', 'content']
)

# 설명 생성용 프롬프트
EXPLANATION_PROMPT = PromptTemplate(
    template="""다음 {subject} 개념에 대해 중학교 1학년 수준으로 설명해주세요.

개념: {concept}
맥락: {context}

설명 조건:
1. 쉽고 이해하기 쉬운 언어 사용
2. 구체적인 예시 포함
3. 단계적 설명
4. 실생활 연관성 강조

설명:""",
    required_variables=['subject', 'concept', 'context']
)

# 품질 평가용 프롬프트
QUALITY_ASSESSMENT_PROMPT = PromptTemplate(
    template="""당신은 AI가 생성한 교육용 문제를 평가하는 전문 평가자입니다.
주어진 원본 컨텍스트와 생성된 문제를 바탕으로, 다음 기준에 따라 품질을 평가해주세요.

---
### 원본 컨텍스트
{source_context}
---
### 생성된 문제 (JSON 형식)
{question_json}
---

### 평가 기준
1.  **관련성 (Relevance)**: 문제가 원본 컨텍스트의 핵심 내용을 다루고 있습니까?
2.  **명확성 (Clarity)**: 질문이 중학생 수준에서 명확하고 모호함 없이 이해할 수 있습니까?
3.  **정확성 (Correctness)**: 문제의 정답과 해설이 사실에 근거하여 정확합니까?
4.  **선택지 타당성 (Plausibility of Distractors)**: 오답 선택지가 매력적이고 그럴듯하여, 단순히 정답을 추측하기 어렵게 만듭니까?
5.  **난이도 일치성 (Difficulty Alignment)**: 문제의 실제 체감 난이도가 명시된 난이도와 일치합니까?

### 출력 형식 (오직 JSON 객체만 반환)
각 항목에 대해 1(매우 나쁨)부터 5(매우 좋음)까지의 점수와 구체적인 평가 이유를 포함하여 아래 JSON 형식으로만 응답해주세요.

```json
{{
    "scores": {{
        "relevance": {{
            "score": 정수(1-5),
            "reason": "평가 이유"
        }},
        "clarity": {{
            "score": 정수(1-5),
            "reason": "평가 이유"
        }},
        "correctness": {{
            "score": 정수(1-5),
            "reason": "평가 이유"
        }},
        "distractor_plausibility": {{
            "score": 정수(1-5),
            "reason": "평가 이유"
        }},
        "difficulty_alignment": {{
            "score": 정수(1-5),
            "reason": "평가 이유"
        }}
    }},
    "overall_score": 평균 점수(소수점 첫째 자리까지),
    "is_usable": 사용 가능 여부 (true/false, 평균 3.5 이상일 때 true),
    "summary": "전반적인 평가 요약 및 개선 제안"
}}
```
""",
    required_variables=['source_context', 'question_json']
)


class PromptManager:
    """프롬프트 관리 클래스"""

    def __init__(self):
        self.templates = {
            'question_generation': QUESTION_GENERATION_PROMPT,
            'math_question': MATH_QUESTION_PROMPT,
            'science_question': SCIENCE_QUESTION_PROMPT,
            'context_summary': CONTEXT_SUMMARY_PROMPT,
            'question_validation': QUESTION_VALIDATION_PROMPT,
            'keyword_extraction': KEYWORD_EXTRACTION_PROMPT,
            'explanation': EXPLANATION_PROMPT,
            'quality_assessment': QUALITY_ASSESSMENT_PROMPT
        }

    def get_template(self, name: str) -> PromptTemplate:
        """템플릿 이름으로 프롬프트 템플릿 가져오기"""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        return self.templates[name]

    def generate_prompt(self, template_name: str, **kwargs) -> str:
        """템플릿으로 프롬프트 생성"""
        template = self.get_template(template_name)
        return template.format(**kwargs)

    def get_subject_specific_template(self, subject: str) -> str:
        """과목별 특화 템플릿 선택"""
        subject_lower = subject.lower()
        if subject_lower in ['수학', 'math', 'mathematics']:
            return 'math_question'
        elif subject_lower in ['과학', 'science']:
            return 'science_question'
        else:
            return 'question_generation'

    def validate_template_variables(self, template_name: str, **kwargs) -> bool:
        """템플릿 변수 유효성 검사"""
        template = self.get_template(template_name)
        return template.validate_variables(**kwargs)

    def add_custom_template(self, name: str, template: str, required_variables: list = None):
        """커스텀 템플릿 추가"""
        self.templates[name] = PromptTemplate(template, required_variables)

    def list_templates(self) -> list:
        """사용 가능한 템플릿 목록"""
        return list(self.templates.keys())


# 전역 프롬프트 매니저 인스턴스
prompt_manager = PromptManager()


def get_question_prompt(subject: str, unit: str, difficulty: str, context: str) -> str:
    """문제 생성용 프롬프트 생성"""
    template_name = prompt_manager.get_subject_specific_template(subject)
    return prompt_manager.generate_prompt(
        template_name,
        subject=subject,
        unit=unit,
        difficulty=difficulty,
        context=context
    )


def get_validation_prompt(question_data: dict) -> str:
    """문제 검증용 프롬프트 생성"""
    question_json = json.dumps(question_data, ensure_ascii=False, indent=2)
    return prompt_manager.generate_prompt(
        'question_validation',
        question_json=question_json
    )


def get_summary_prompt(content: str, subject: str, unit: str, length: str = "medium") -> str:
    """요약용 프롬프트 생성"""
    return prompt_manager.generate_prompt(
        'context_summary',
        content=content,
        subject=subject,
        unit=unit,
        length=length
    )


def get_keyword_extraction_prompt(content: str, subject: str) -> str:
    """키워드 추출용 프롬프트 생성"""
    return prompt_manager.generate_prompt(
        'keyword_extraction',
        content=content,
        subject=subject
    )


def get_explanation_prompt(concept: str, subject: str, context: str) -> str:
    """설명 생성용 프롬프트 생성"""
    return prompt_manager.generate_prompt(
        'explanation',
        concept=concept,
        subject=subject,
        context=context
    )


def get_quality_assessment_prompt() -> PromptTemplate:
    """품질 평가용 프롬프트 템플릿 가져오기"""
    return prompt_manager.get_template('quality_assessment')