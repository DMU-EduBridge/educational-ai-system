from typing import List, Optional, Literal
import time
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer
import tiktoken


class EmbeddingsManager:
    """임베딩 생성 및 관리 (Google Generative AI 또는 로컬 모델 사용)"""

    def __init__(
        self, 
        model_name: str = "models/embedding-001", 
        api_key: Optional[str] = None,
        provider: Literal["google", "local"] = "local"
    ):
        """
        EmbeddingsManager 초기화

        Args:
            model_name: 임베딩 모델명
                - Google: "models/embedding-001", "models/text-embedding-004" 등
                - Local: "jhgan/ko-sroberta-multitask", "sentence-transformers/all-MiniLM-L6-v2" 등
            api_key: Google API 키 (provider="google"인 경우 필요)
            provider: 임베딩 제공자 ("google" 또는 "local")
        """
        self.model_name = model_name
        self.provider = provider
        
        if provider == "google":
            self.client = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=api_key
            )
            self.is_local = False
        else:  # provider == "local"
            # 한국어 최적화 로컬 모델 사용 (무료, 할당량 없음)
            default_local_model = "jhgan/ko-sroberta-multitask"
            self.client = SentenceTransformer(model_name or default_local_model)
            self.is_local = True
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Using local embedding model: {model_name or default_local_model}")
        
        # 토큰 카운터 (대략적 추정용)
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoding = None
            
        self.logger = logging.getLogger(__name__)

        # API 제한 (로컬 모델은 제한 없음)
        if self.is_local:
            self.max_tokens_per_minute = float('inf')
            self.max_requests_per_minute = float('inf')
            self.batch_size = 32  # 로컬 모델은 메모리에 따라 조정
        else:
            self.max_tokens_per_minute = 1000000
            self.max_requests_per_minute = 3000
            self.batch_size = 100

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트 리스트에 대한 임베딩 생성 (배치 처리)

        Args:
            texts: 임베딩을 생성할 텍스트 리스트

        Returns:
            List[List[float]]: 임베딩 벡터 리스트
        """
        if not texts:
            return []

        if self.is_local:
            # 로컬 모델 사용 (sentence-transformers)
            try:
                embeddings = self.client.encode(
                    texts, 
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                return embeddings.tolist()
            except Exception as e:
                self.logger.error(f"Error generating embeddings with local model: {str(e)}")
                raise

        # Google API 사용
        all_embeddings = []

        # 배치 단위로 처리
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            # 토큰 수 확인
            total_tokens = sum(self._count_tokens(text) for text in batch)
            self.logger.info(f"Processing batch {i//self.batch_size + 1}: {len(batch)} texts, {total_tokens} tokens")

            try:
                # Langchain의 embed_documents 메서드 사용
                batch_embeddings = self.client.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)

                # API 속도 제한 방지를 위한 대기
                if len(batch) == self.batch_size:
                    time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error generating embeddings for batch: {str(e)}")
                raise

        return all_embeddings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate_single_embedding(self, text: str) -> List[float]:
        """
        단일 텍스트에 대한 임베딩 생성

        Args:
            text: 임베딩을 생성할 텍스트

        Returns:
            List[float]: 임베딩 벡터
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")

        if self.is_local:
            # 로컬 모델 사용
            try:
                embedding = self.client.encode(
                    text,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                return embedding.tolist()
            except Exception as e:
                self.logger.error(f"Error generating single embedding with local model: {str(e)}")
                raise

        # Google API 사용
        try:
            # Langchain의 embed_query 메서드 사용
            embedding = self.client.embed_query(text)
            return embedding

        except Exception as e:
            self.logger.error(f"Error generating single embedding: {str(e)}")
            raise

    def estimate_cost(self, texts: List[str]) -> dict:
        """
        임베딩 생성 비용 추정

        Args:
            texts: 텍스트 리스트

        Returns:
            dict: 비용 정보
        """
        if not texts:
            return {
                'total_tokens': 0,
                'total_cost': 0.0,
                'num_texts': 0,
                'model': self.model_name,
                'provider': self.provider
            }

        total_tokens = sum(self._count_tokens(text) for text in texts)

        if self.is_local:
            # 로컬 모델은 무료
            return {
                'total_tokens': total_tokens,
                'total_cost': 0.0,
                'num_texts': len(texts),
                'cost_per_1k_tokens': 0.0,
                'model': self.model_name,
                'provider': 'local',
                'note': 'Local model - Free, no API costs, no quota limits'
            }

        # Google Embedding API 가격: 무료 또는 매우 저렴 (2024년 기준 무료)
        cost_per_1k_tokens = 0.0  # 현재 무료
        estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens

        return {
            'total_tokens': total_tokens,
            'total_cost': round(estimated_cost, 6),
            'num_texts': len(texts),
            'cost_per_1k_tokens': cost_per_1k_tokens,
            'model': self.model_name,
            'provider': 'google'
        }

    def _count_tokens(self, text: str) -> int:
        """
        텍스트의 토큰 수 계산

        Args:
            text: 입력 텍스트

        Returns:
            int: 토큰 수
        """
        try:
            if self.encoding:
                return len(self.encoding.encode(text))
            else:
                # 대략적인 추정치 (1 토큰 ≈ 4 문자)
                return len(text) // 4
        except Exception as e:
            self.logger.warning(f"Error counting tokens: {str(e)}")
            # 대략적인 추정치 (1 토큰 ≈ 4 문자)
            return len(text) // 4

    def validate_text_length(self, text: str) -> bool:
        """
        텍스트가 모델의 최대 토큰 길이를 초과하는지 확인

        Args:
            text: 확인할 텍스트

        Returns:
            bool: 유효한 길이인지 여부
        """
        # Google Embedding API의 최대 토큰 길이: 약 20,000 토큰
        max_tokens = 20000
        token_count = self._count_tokens(text)

        return token_count <= max_tokens

    def split_long_text(self, text: str, max_tokens: int = 19000) -> List[str]:
        """
        긴 텍스트를 토큰 제한에 맞게 분할

        Args:
            text: 분할할 텍스트
            max_tokens: 최대 토큰 수 (Google은 약 20,000)

        Returns:
            List[str]: 분할된 텍스트 리스트
        """
        if self._count_tokens(text) <= max_tokens:
            return [text]

        # 문장 단위로 분할하여 토큰 제한 맞추기
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            test_chunk = current_chunk + sentence + ". " if current_chunk else sentence + ". "

            if self._count_tokens(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks