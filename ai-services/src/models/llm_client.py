from typing import Optional, Dict, Any, List
import openai
import tiktoken
import time
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime


class LLMClient:
    """OpenAI API 클라이언트"""

    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 1000):
        """
        LLMClient 초기화

        Args:
            model_name: OpenAI 모델명
            api_key: OpenAI API 키
            temperature: 응답의 창의성 (0.0-2.0)
            max_tokens: 최대 토큰 수
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
        self.logger = logging.getLogger(__name__)

        # 토큰 카운터 초기화
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 기본 인코딩

        # 사용량 추적
        self.usage_stats = {
            'total_requests': 0,
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_cost_usd': 0.0,
            'requests_by_hour': {},
            'last_request_time': None
        }

        # 모델별 가격 정보 (per 1K tokens)
        self.pricing = {
            'gpt-3.5-turbo': {'prompt': 0.0015, 'completion': 0.002},
            'gpt-3.5-turbo-16k': {'prompt': 0.003, 'completion': 0.004},
            'gpt-4': {'prompt': 0.03, 'completion': 0.06},
            'gpt-4-32k': {'prompt': 0.06, 'completion': 0.12},
            'gpt-4-turbo-preview': {'prompt': 0.01, 'completion': 0.03},
            'gpt-4o': {'prompt': 0.005, 'completion': 0.015}
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate_response(self,
                         prompt: str,
                         max_tokens: Optional[int] = None,
                         temperature: Optional[float] = None,
                         system_message: Optional[str] = None) -> str:
        """
        프롬프트에 대한 응답 생성

        Args:
            prompt: 입력 프롬프트
            max_tokens: 최대 토큰 수 (None시 기본값 사용)
            temperature: 창의성 수준 (None시 기본값 사용)
            system_message: 시스템 메시지

        Returns:
            str: 생성된 응답
        """
        try:
            # 매개변수 설정
            actual_max_tokens = max_tokens or self.max_tokens
            actual_temperature = temperature if temperature is not None else self.temperature

            # 메시지 구성
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            # 토큰 수 추정
            prompt_tokens = self._count_messages_tokens(messages)

            # API 호출
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=actual_max_tokens,
                temperature=actual_temperature,
                n=1,
                stop=None
            )

            # 응답 처리
            generated_text = response.choices[0].message.content
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            # 사용량 업데이트
            self._update_usage_stats(prompt_tokens, completion_tokens, total_tokens)

            response_time = time.time() - start_time
            self.logger.info(
                f"Generated response: {prompt_tokens} prompt + {completion_tokens} completion tokens "
                f"in {response_time:.2f}s"
            )

            return generated_text

        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise

    def generate_structured_response(self,
                                   prompt: str,
                                   response_format: str = "json",
                                   max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        구조화된 응답 생성 (JSON 등)

        Args:
            prompt: 입력 프롬프트
            response_format: 응답 형식 ("json" 등)
            max_tokens: 최대 토큰 수

        Returns:
            Dict[str, Any]: 파싱된 구조화된 응답
        """
        try:
            if response_format == "json":
                system_message = (
                    "You must respond with valid JSON only. "
                    "Do not include any explanations or additional text outside the JSON."
                )
                response_text = self.generate_response(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    system_message=system_message
                )

                # JSON 파싱
                import json
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError as e:
                    # JSON 파싱 실패 시 재시도
                    self.logger.warning(f"JSON parsing failed: {str(e)}")
                    # 간단한 JSON 수정 시도
                    cleaned_response = self._clean_json_response(response_text)
                    return json.loads(cleaned_response)

            else:
                raise ValueError(f"Unsupported response format: {response_format}")

        except Exception as e:
            self.logger.error(f"Error generating structured response: {str(e)}")
            raise

    def estimate_tokens(self, text: str) -> int:
        """
        텍스트의 토큰 수 추정

        Args:
            text: 입력 텍스트

        Returns:
            int: 추정 토큰 수
        """
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            self.logger.warning(f"Error counting tokens: {str(e)}")
            # 대략적인 추정치 (1 토큰 ≈ 4 문자)
            return len(text) // 4

    def estimate_cost(self, prompt: str, estimated_completion_tokens: int = 500) -> Dict[str, float]:
        """
        요청 비용 추정

        Args:
            prompt: 입력 프롬프트
            estimated_completion_tokens: 예상 완료 토큰 수

        Returns:
            Dict[str, float]: 비용 정보
        """
        prompt_tokens = self.estimate_tokens(prompt)

        pricing = self.pricing.get(self.model_name, self.pricing['gpt-3.5-turbo'])

        prompt_cost = (prompt_tokens / 1000) * pricing['prompt']
        completion_cost = (estimated_completion_tokens / 1000) * pricing['completion']
        total_cost = prompt_cost + completion_cost

        return {
            'prompt_tokens': prompt_tokens,
            'estimated_completion_tokens': estimated_completion_tokens,
            'prompt_cost_usd': round(prompt_cost, 6),
            'completion_cost_usd': round(completion_cost, 6),
            'total_cost_usd': round(total_cost, 6),
            'model': self.model_name
        }

    def track_usage(self) -> Dict[str, Any]:
        """
        사용량 통계 반환

        Returns:
            Dict[str, Any]: 사용량 정보
        """
        return {
            'model': self.model_name,
            'total_requests': self.usage_stats['total_requests'],
            'total_prompt_tokens': self.usage_stats['total_prompt_tokens'],
            'total_completion_tokens': self.usage_stats['total_completion_tokens'],
            'total_tokens': self.usage_stats['total_prompt_tokens'] + self.usage_stats['total_completion_tokens'],
            'total_cost_usd': round(self.usage_stats['total_cost_usd'], 6),
            'average_tokens_per_request': (
                (self.usage_stats['total_prompt_tokens'] + self.usage_stats['total_completion_tokens'])
                / max(self.usage_stats['total_requests'], 1)
            ),
            'last_request_time': self.usage_stats['last_request_time']
        }

    def reset_usage_stats(self):
        """사용량 통계 초기화"""
        self.usage_stats = {
            'total_requests': 0,
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_cost_usd': 0.0,
            'requests_by_hour': {},
            'last_request_time': None
        }
        self.logger.info("Usage statistics reset")

    def _count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        메시지 리스트의 토큰 수 계산

        Args:
            messages: 메시지 리스트

        Returns:
            int: 토큰 수
        """
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # 메시지당 기본 토큰
            for key, value in message.items():
                num_tokens += len(self.encoding.encode(value))
                if key == "name":
                    num_tokens += -1  # name 필드는 1 토큰 감소

        num_tokens += 2  # 어시스턴트 응답을 위한 준비 토큰
        return num_tokens

    def _update_usage_stats(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        """
        사용량 통계 업데이트

        Args:
            prompt_tokens: 프롬프트 토큰 수
            completion_tokens: 완료 토큰 수
            total_tokens: 총 토큰 수
        """
        # 기본 통계 업데이트
        self.usage_stats['total_requests'] += 1
        self.usage_stats['total_prompt_tokens'] += prompt_tokens
        self.usage_stats['total_completion_tokens'] += completion_tokens

        # 비용 계산
        pricing = self.pricing.get(self.model_name, self.pricing['gpt-3.5-turbo'])
        prompt_cost = (prompt_tokens / 1000) * pricing['prompt']
        completion_cost = (completion_tokens / 1000) * pricing['completion']
        self.usage_stats['total_cost_usd'] += prompt_cost + completion_cost

        # 시간 기록
        now = datetime.now()
        self.usage_stats['last_request_time'] = now.isoformat()

        # 시간당 요청 수 추적
        hour_key = now.strftime('%Y-%m-%d %H')
        self.usage_stats['requests_by_hour'][hour_key] = (
            self.usage_stats['requests_by_hour'].get(hour_key, 0) + 1
        )

    def _clean_json_response(self, response: str) -> str:
        """
        JSON 응답 정리

        Args:
            response: 원본 응답

        Returns:
            str: 정리된 JSON 문자열
        """
        # 코드 블록 제거
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        if response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]

        # 앞뒤 공백 제거
        response = response.strip()

        return response