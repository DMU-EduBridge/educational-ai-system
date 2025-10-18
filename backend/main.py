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
project_root = Path(__file__).resolve().parent.parent
ai_services_path = project_root / 'ai-services'
if str(ai_services_path) not in sys.path:
    sys.path.insert(0, str(ai_services_path))

try:
    # ai-services의 RAGPipeline 임포트
    from src.main import RAGPipeline
    from src.utils.logger import get_logger
    from src.analysis.student_analyzer import StudentAnalyzer
except ImportError as e:
    print(f"Error importing from ai-services: {e}")
    sys.exit(1)

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title="Educational AI System - Question Generation API",
    description="AI를 활용하여 교육용 문제를 생성하는 API입니다.",
    version="1.0.0",
)

# CORS 설정
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
analyzer = None

@app.on_event("startup")
def startup_event():
    """애플리케이션 시작 시 RAG 파이프라인을 초기화합니다."""
    global pipeline, analyzer
    try:
        logger.info("Initializing RAG Pipeline...")
        pipeline = RAGPipeline()
        analyzer = StudentAnalyzer(llm_client=pipeline.llm_client)
        logger.info("RAG Pipeline and Student Analyzer initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        pipeline = None
        analyzer = None

# 요청 본문을 위한 Pydantic 모델
class QuestionRequest(BaseModel):
    subject: str = Field(..., description="문제 과목", example="수학")
    unit: str = Field(..., description="세부 단원", example="일차함수")
    difficulty: str = Field("medium", description="문제 난이도", example="medium")
    count: int = Field(1, gt=0, le=10, description="생성할 문제 수")

class AnalysisRequest(BaseModel):
    user_id: str = Field(..., description="분석할 학생의 ID", example="user_1234")

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
        logger.warning(f"Value error during question generation: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"An unexpected error occurred during question generation: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while generating questions.")

@app.post("/analyze-student-performance", summary="학생 학습 성과 분석")
async def analyze_student_performance_endpoint(request: AnalysisRequest) -> Dict[str, str]:
    """
    특정 학생의 문제 풀이 로그를 분석하여 종합적인 학습 리포트를 생성합니다.
    """
    if not analyzer:
        raise HTTPException(status_code=503, detail="Student Analyzer is not available.")

    try:
        logger.info(f"Received request to analyze performance for user {request.user_id}")
        
        report = analyzer.analyze(request.user_id)
        
        if "해당 사용자에 대한 학습 로그를 찾을 수 없습니다." in report or "error" in report:
            raise HTTPException(status_code=404, detail=report)

        logger.info(f"Successfully generated analysis report for user {request.user_id}.")
        return {"report": report}

    except Exception as e:
        logger.error(f"An unexpected error occurred during analysis: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while analyzing performance.")

# 서버 실행을 위한 uvicorn 명령어 (터미널에서 실행):
# uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
