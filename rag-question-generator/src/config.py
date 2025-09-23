"""
Configuration Management - 설정 관리
환경 변수와 설정을 관리합니다.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """애플리케이션 설정"""

    # OpenAI 설정
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"
    openai_embedding_model: str = "text-embedding-ada-002"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 1000

    # RAG 설정
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 5

    # ChromaDB 설정
    chroma_db_path: str = "./ai-services/rag-question-generator/data/vector_db"
    chroma_collection_name: str = "document_chunks"

    # 문제 생성 설정
    default_question_count: int = 10
    default_difficulty_mix: str = "balanced"

    # 로깅 설정
    log_level: str = "INFO"
    debug: bool = False

    class Config:
        env_file = ".env"
        env_prefix = ""
        case_sensitive = False

    @validator("openai_api_key")
    def validate_api_key(cls, v):
        if not v or not v.startswith("sk-"):
            raise ValueError("OpenAI API 키가 올바르지 않습니다. 'sk-'로 시작해야 합니다.")
        return v

    @validator("chunk_size")
    def validate_chunk_size(cls, v):
        if v < 100 or v > 4000:
            raise ValueError("chunk_size는 100-4000 사이여야 합니다.")
        return v

    @validator("chunk_overlap")
    def validate_chunk_overlap(cls, v):
        if v < 0 or v > 1000:
            raise ValueError("chunk_overlap은 0-1000 사이여야 합니다.")
        return v

    @validator("openai_temperature")
    def validate_temperature(cls, v):
        if v < 0.0 or v > 2.0:
            raise ValueError("temperature는 0.0-2.0 사이여야 합니다.")
        return v

    @validator("retrieval_k")
    def validate_retrieval_k(cls, v):
        if v < 1 or v > 20:
            raise ValueError("retrieval_k는 1-20 사이여야 합니다.")
        return v

    def validate_api_key_connection(self) -> bool:
        """API 키 연결 테스트"""
        try:
            import openai
            client = openai.OpenAI(api_key=self.openai_api_key)

            # 간단한 API 호출로 테스트
            response = client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            print(f"❌ API 키 연결 실패: {str(e)}")
            return False

    def to_dict(self) -> dict:
        """설정을 딕셔너리로 변환 (API 키 제외)"""
        return {
            "openai_model": self.openai_model,
            "openai_embedding_model": self.openai_embedding_model,
            "openai_temperature": self.openai_temperature,
            "openai_max_tokens": self.openai_max_tokens,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_k": self.retrieval_k,
            "chroma_db_path": self.chroma_db_path,
            "chroma_collection_name": self.chroma_collection_name,
            "default_question_count": self.default_question_count,
            "default_difficulty_mix": self.default_difficulty_mix,
            "log_level": self.log_level,
            "debug": self.debug
        }


def get_settings() -> Settings:
    """설정 인스턴스 반환"""
    return Settings()


def check_environment() -> dict:
    """환경 설정 상태 확인"""
    status = {
        "env_file_exists": False,
        "api_key_set": False,
        "api_key_valid": False,
        "vector_db_path_exists": False,
        "issues": []
    }

    # .env 파일 확인
    env_file = Path(".env")
    if env_file.exists():
        status["env_file_exists"] = True
    else:
        status["issues"].append(".env 파일이 없습니다")

    # API 키 확인
    try:
        settings = get_settings()
        status["api_key_set"] = bool(settings.openai_api_key)

        if status["api_key_set"]:
            status["api_key_valid"] = settings.validate_api_key_connection()
        else:
            status["issues"].append("OPENAI_API_KEY가 설정되지 않았습니다")

    except Exception as e:
        status["issues"].append(f"설정 로드 실패: {str(e)}")

    # 벡터 DB 경로 확인
    try:
        settings = get_settings()
        db_path = Path(settings.chroma_db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        status["vector_db_path_exists"] = True
    except Exception as e:
        status["issues"].append(f"벡터 DB 경로 생성 실패: {str(e)}")

    return status


def setup_environment() -> bool:
    """환경 설정 초기화"""
    try:
        print("🔧 환경 설정 초기화 중...")

        # 필요한 디렉토리 생성
        directories = [
            "./ai-services/rag-question-generator/data/input",
            "./ai-services/rag-question-generator/data/output",
            "./ai-services/rag-question-generator/data/vector_db"
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"📁 디렉토리 생성: {directory}")

        # 환경 상태 확인
        status = check_environment()

        print("\n📊 환경 설정 상태:")
        print(f"  .env 파일: {'✅' if status['env_file_exists'] else '❌'}")
        print(f"  API 키 설정: {'✅' if status['api_key_set'] else '❌'}")
        print(f"  API 키 유효성: {'✅' if status['api_key_valid'] else '❌'}")
        print(f"  벡터 DB 경로: {'✅' if status['vector_db_path_exists'] else '❌'}")

        if status["issues"]:
            print("\n⚠️ 발견된 문제:")
            for issue in status["issues"]:
                print(f"  - {issue}")

        success = (status["env_file_exists"] and
                  status["api_key_set"] and
                  status["api_key_valid"] and
                  status["vector_db_path_exists"])

        if success:
            print("\n✅ 환경 설정이 완료되었습니다!")
        else:
            print("\n❌ 환경 설정에 문제가 있습니다. .env 파일과 API 키를 확인하세요.")

        return success

    except Exception as e:
        print(f"❌ 환경 설정 실패: {str(e)}")
        return False