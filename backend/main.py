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
    from src.chatbot.tutor import ChatbotTutor
except ImportError as e:
    print(f"Error importing from ai-services: {e}")
    sys.exit(1)

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title="Educational AI System - API",
    description="AI를 활용하여 교육용 문제를 생성하고, 학생 맞춤형 학습을 제공하는 API입니다.",
    version="1.2.0",
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

@app.on_event("startup")
def startup_event():
    """애플리케이션 시작 시 RAG 파이프라인을 초기화합니다."""
    global pipeline
    try:
        logger.info("Initializing RAG Pipeline...")
        pipeline = RAGPipeline()
        logger.info("RAG Pipeline initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        pipeline = None

# --- Pydantic 모델 --- #
class QuestionRequest(BaseModel):
    subject: str = Field(..., description="문제 과목", example="수학")
    unit: str = Field(..., description="세부 단원", example="일차함수")
    difficulty: str = Field("medium", description="문제 난이도", example="medium")
    count: int = Field(1, gt=0, le=10, description="생성할 문제 수")

class ChatMessage(BaseModel):
    user_id: str = Field(..., description="학생의 ID", example="user_1234")
    user_message: str = Field(..., description="사용자의 메시지", example="개념을 설명해줄래?")
    history: List[Dict[str, str]] = Field([], description="이전 대화 기록")

class ChatResponse(BaseModel):
    ai_response: str
    updated_history: List[Dict[str, str]]

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

@app.post("/chat/message", summary="챗봇과 메시지 주고받기 (REST)")
async def chat_message_endpoint(request: ChatMessage) -> ChatResponse:
    if not pipeline:
        raise HTTPException(status_code=503, detail="Core services are not available.")
    
    try:
        tutor = ChatbotTutor(request.user_id, pipeline.llm_client)
        
        if not tutor.analysis_context:
            raise HTTPException(status_code=404, detail=f"No report found for user {request.user_id}. A weekly report must be generated first.")

        new_history = request.history + [{"role": "user", "content": request.user_message}]
        ai_response = tutor.get_response(request.user_message, new_history)
        new_history.append({"role": "assistant", "content": ai_response})
        
        return ChatResponse(ai_response=ai_response, updated_history=new_history)

    except Exception as e:
        logger.error(f"Error in REST chat for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during chat.")

# --- WebSocket 엔드포인트 --- #
@app.websocket("/ws/chat/{user_id}")
async def websocket_chat_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    
    if not pipeline:
        await websocket.close(code=1011, reason="Core services are not available.")
        return

    try:
        tutor = ChatbotTutor(user_id, pipeline.llm_client)
        initial_message, history = tutor.start_session()
        
        if not tutor.analysis_context:
            await websocket.send_text(f"No report found for user {user_id}. Please wait for the weekly report.")
            await websocket.close()
            return

        await websocket.send_text(initial_message)

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
