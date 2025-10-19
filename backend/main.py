import sys
import os
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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
    from src.chatbot.tutor import ChatbotTutor
except ImportError as e:
    print(f"Error importing from ai-services: {e}")
    sys.exit(1)

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title="Educational AI System - API",
    description="AI를 활용하여 교육용 문제를 생성하고, 학생 맞춤형 학습을 제공하는 API입니다.",
    version="1.1.0",
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

# --- Pydantic 모델 --- #
class QuestionRequest(BaseModel):
    subject: str = Field(..., description="문제 과목", example="수학")
    unit: str = Field(..., description="세부 단원", example="일차함수")
    difficulty: str = Field("medium", description="문제 난이도", example="medium")
    count: int = Field(1, gt=0, le=10, description="생성할 문제 수")

class AnalysisRequest(BaseModel):
    user_id: str = Field(..., description="분석할 학생의 ID", example="user_1234")

# --- REST API 엔드포인트 --- #
@app.get("/", summary="API 상태 확인")
def read_root():
    """API 서버의 기본 상태를 확인하는 엔드포인트입니다."""
    if pipeline:
        return {"status": "ok", "message": "Welcome to the Educational AI System API!"}
    return {"status": "error", "message": "RAG Pipeline is not initialized."}


@app.post("/generate-question", summary="새로운 문제 생성")
async def generate_question_endpoint(request: QuestionRequest) -> List[Dict[str, Any]]:
    if not pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline is not available.")
    try:
        return pipeline.generate_questions(
            subject=request.subject,
            unit=request.unit,
            difficulty=request.difficulty,
            count=request.count,
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during question generation: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")

@app.post("/analyze-student-performance", summary="학생 학습 성과 분석")
async def analyze_student_performance_endpoint(request: AnalysisRequest) -> Dict[str, Any]:
    if not analyzer:
        raise HTTPException(status_code=503, detail="Student Analyzer is not available.")
    try:
        report = analyzer.analyze(request.user_id)
        if "error" in report:
            raise HTTPException(status_code=404, detail=report["error"])
        return report
    except Exception as e:
        logger.error(f"An unexpected error occurred during analysis: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")

# --- WebSocket 엔드포인트 --- #
@app.websocket("/ws/chat/{user_id}")
async def websocket_chat_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    
    if not pipeline:
        await websocket.close(code=1011, reason="Core services are not available.")
        return

    tutor = ChatbotTutor(user_id, pipeline.llm_client)
    
    try:
        # 세션 시작 및 첫 메시지 전송
        initial_message, history = tutor.start_session()
        await websocket.send_text(initial_message)

        # 대화 루프
        while True:
            user_message = await websocket.receive_text()
            history.append({"role": "user", "content": user_message})
            
            ai_response = tutor.get_response(user_message, history)
            history.append({"role": "assistant", "content": ai_response})
            
            await websocket.send_text(ai_response)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user: {user_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket chat for user {user_id}: {e}")
        await websocket.close(code=1011, reason="An internal error occurred.")

# 서버 실행을 위한 uvicorn 명령어 (터미널에서 실행):
# uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
