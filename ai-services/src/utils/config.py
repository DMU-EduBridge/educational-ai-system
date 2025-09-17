from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import os
from pathlib import Path


class Settings(BaseSettings):
    """애플리케이션 설정"""

    # OpenAI API 설정
    openai_api_key: str = Field(..., description="OpenAI API Key")
    openai_model: str = Field(default="gpt-3.5-turbo", description="OpenAI LLM Model")
    openai_embedding_model: str = Field(default="text-embedding-ada-002", description="OpenAI Embedding Model")
    openai_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM Temperature")
    openai_max_tokens: int = Field(default=1000, ge=1, le=4000, description="Max tokens for LLM responses")

    # ChromaDB 설정
    chroma_db_path: str = Field(default="./data/vector_db", description="ChromaDB persist directory")
    chroma_collection_name: str = Field(default="textbook_embeddings", description="ChromaDB collection name")

    # 텍스트 처리 설정
    chunk_size: int = Field(default=1000, ge=100, le=4000, description="Text chunk size")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="Text chunk overlap")
    max_context_length: int = Field(default=3000, ge=500, le=8000, description="Max context length for RAG")

    # 검색 설정
    retrieval_k: int = Field(default=3, ge=1, le=10, description="Number of documents to retrieve")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold for retrieval")

    # 로깅 설정
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")

    # 개발 모드
    debug: bool = Field(default=False, description="Debug mode")
    verbose: bool = Field(default=False, description="Verbose output")

    # 캐시 설정
    enable_cache: bool = Field(default=True, description="Enable embedding cache")
    cache_dir: str = Field(default="./data/cache", description="Cache directory")

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = False
        extra = "ignore"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 디렉토리 생성
        self._create_directories()

    def _create_directories(self):
        """필요한 디렉토리들을 생성"""
        directories = [
            self.chroma_db_path,
            self.cache_dir,
            "./data/sample_textbooks",
            "./logs" if self.log_file else None
        ]

        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)

    @property
    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return not self.debug

    def get_openai_config(self) -> dict:
        """OpenAI 설정 반환"""
        return {
            'api_key': self.openai_api_key,
            'model': self.openai_model,
            'embedding_model': self.openai_embedding_model,
            'temperature': self.openai_temperature,
            'max_tokens': self.openai_max_tokens
        }

    def get_chroma_config(self) -> dict:
        """ChromaDB 설정 반환"""
        return {
            'persist_directory': self.chroma_db_path,
            'collection_name': self.chroma_collection_name
        }

    def get_text_processing_config(self) -> dict:
        """텍스트 처리 설정 반환"""
        return {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'max_context_length': self.max_context_length
        }

    def get_retrieval_config(self) -> dict:
        """검색 설정 반환"""
        return {
            'k': self.retrieval_k,
            'similarity_threshold': self.similarity_threshold
        }

    def validate_api_key(self) -> bool:
        """API 키 유효성 검사"""
        if not self.openai_api_key or self.openai_api_key == "your_openai_api_key_here":
            return False

        # 기본적인 형식 검사
        if not self.openai_api_key.startswith('sk-'):
            return False

        return len(self.openai_api_key) > 20

    def update_setting(self, key: str, value) -> bool:
        """설정 값 업데이트"""
        try:
            if hasattr(self, key):
                setattr(self, key, value)
                return True
            return False
        except Exception:
            return False

    def to_dict(self) -> dict:
        """설정을 딕셔너리로 변환 (API 키 제외)"""
        config_dict = self.dict()
        # 민감한 정보 마스킹
        if 'openai_api_key' in config_dict:
            config_dict['openai_api_key'] = f"sk-...{config_dict['openai_api_key'][-4:]}"
        return config_dict


# 글로벌 설정 인스턴스
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """글로벌 설정 인스턴스 반환"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """설정 재로드"""
    global _settings
    _settings = Settings()
    return _settings


def update_global_setting(key: str, value) -> bool:
    """글로벌 설정 업데이트"""
    settings = get_settings()
    return settings.update_setting(key, value)


# 환경별 설정 함수들
def get_development_settings() -> Settings:
    """개발 환경 설정"""
    return Settings(
        debug=True,
        verbose=True,
        log_level="DEBUG",
        chunk_size=500,  # 개발시 작은 청크
        retrieval_k=2
    )


def get_production_settings() -> Settings:
    """프로덕션 환경 설정"""
    return Settings(
        debug=False,
        verbose=False,
        log_level="INFO",
        chunk_size=1000,
        retrieval_k=3
    )


def get_test_settings() -> Settings:
    """테스트 환경 설정"""
    return Settings(
        debug=True,
        verbose=False,
        log_level="WARNING",
        chroma_db_path="./test_data/vector_db",
        cache_dir="./test_data/cache",
        chunk_size=200,  # 테스트용 작은 청크
        retrieval_k=1
    )