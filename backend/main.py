"""
FastAPI Backend for Educational AI System
"""
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 프로젝트 루트를 시스템 경로에 추가
# 이렇게 함으로써 'ai-services' 모듈을 찾을 수 있습니다.
project_root = Path(__file__).resolve().parent.parent
ai_services_path = project_root / 'ai-services'
if str(ai_services_path) not in sys.path:
    sys.path.insert(0, str(ai_services_path))

try:
    # ai-services의 RAGPipeline 임포트
    from src.main import RAGPipeline
    from src.utils.logger import get_logger
except ImportError as e:
    print(f"Error importing from ai-services: {e}")
    # 환경 변수나 경로 설정을 확인하라는 메시지를 포함할 수 있습니다.
    sys.exit(1)

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title="Educational AI System - Question Generation API",
    description="AI를 활용하여 교육용 문제를 생성하는 API입니다.",
    version="1.0.0",
)

# CORS 설정
# 모든 출처에서의 요청을 허용합니다 (개발용).
# 프로덕션 환경에서는 특정 도메인만 허용하도록 수정해야 합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 로거 및 RAG 파이프라인 초기화
logger = get_logger(__name__)
pipeline = None

@app.on_event("startup")
def startup_event():
    """애플리케이션 시작 시 RAG 파이프라인을 초기화합니다."""
    global pipeline
    try:
        logger.info("Initializing RAG Pipeline...")
        pipeline = RAGPipeline()
        logger.info("RAG Pipeline initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Pipeline: {e}")
        # 파이프라인 초기화 실패는 심각한 문제이므로,
        # 애플리케이션을 시작하지 않거나 상태를 "unhealthy"로 설정할 수 있습니다.
        # 여기서는 간단히 에러 로그만 남깁니다.
        pipeline = None

# 요청 본문을 위한 Pydantic 모델
class QuestionRequest(BaseModel):
    subject: str = Field(..., description="문제 과목", example="수학")
    unit: str = Field(..., description="세부 단원", example="일차함수")
    difficulty: str = Field("medium", description="문제 난이도", example="medium")
    count: int = Field(1, gt=0, le=10, description="생성할 문제 수")

@app.get("/", summary="API 상태 확인")
def read_root():
    """API 서버의 기본 상태를 확인하는 엔드포인트입니다."""
    if pipeline:
        return {"status": "ok", "message": "Welcome to the Educational AI System API!"}
    return {"status": "error", "message": "RAG Pipeline is not initialized."}


@app.post("/generate-question", summary="새로운 문제 생성")
async def generate_question_endpoint(request: QuestionRequest) -> List[Dict[str, Any]]:
    """
    주어진 과목, 단원, 난이도에 따라 하나 이상의 새로운 문제를 생성합니다.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline is not available.")

    try:
        logger.info(f"Received request to generate {request.count} question(s) for {request.subject} - {request.unit}")
        
        questions = pipeline.generate_questions(
            subject=request.subject,
            unit=request.unit,
            difficulty=request.difficulty,
            count=request.count,
        )
        
        if not questions:
            raise HTTPException(status_code=404, detail="Could not generate any questions for the given topic.")
            
        logger.info(f"Successfully generated {len(questions)} question(s).")
        return questions

    except ValueError as ve:
        # 특정 값 관련 에러 (예: 컨텍스트 없음)
        logger.warning(f"Value error during question generation: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"An unexpected error occurred during question generation: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while generating questions.")

# 서버 실행을 위한 uvicorn 명령어 (터미널에서 실행):
# uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
