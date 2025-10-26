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
    from src.analysis.student_analyzer import StudentAnalyzer
    from src.models.llm_client import LLMClient
except ImportError as e:
    print(f"Error importing from ai-services: {e}")
    sys.exit(1)

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title="Educational AI System - API",
    description="AI를 활용하여 교육용 문제를 생성하고, 학생 맞춤형 학습을 제공하는 API입니다.",
    version="1.3.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 로거, 파이프라인, 및 대화 기록 저장소 초기화
logger = get_logger(__name__)
pipeline = None
chat_histories: Dict[str, List[Dict[str, str]]] = {}

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

class ProblemResponse(BaseModel):
    """DB Problem 테이블 형식에 맞춘 응답 모델"""
    id: str | None
    title: str
    description: str | None
    content: str
    type: str
    difficulty: str
    subject: str
    gradeLevel: str
    unit: str
    options: List[str]
    correctAnswer: str
    explanation: str | None
    hints: List[str] | None
    tags: List[str] | None
    points: int
    timeLimit: int | None
    isActive: bool
    isAIGenerated: bool
    aiGenerationId: str
    qualityScore: float | None
    reviewStatus: str
    status: str
    modelName: str
    tokensUsed: int | None
    costUsd: float | None
    createdAt: str
    updatedAt: str

class ChatRequest(BaseModel):
    user_id: str = Field(..., description="학생의 ID", example="user_1234")
    user_message: str = Field(..., description="사용자의 메시지", example="개념을 설명해줄래?")

class ChatResponse(BaseModel):
    ai_response: str

class ReportRequest(BaseModel):
    user_id: str = Field(..., description="분석할 학생의 ID", example="user_1234")

class ReportResponse(BaseModel):
    user_id: str
    report_text: str
    weakest_unit: str
    performance_summary: Dict[str, Any]
    generated_at: str

# --- REST API 엔드포인트 --- #
@app.get("/", summary="API 상태 확인")
def read_root():
    """API 서버의 기본 상태를 확인하는 엔드포인트입니다."""
    if pipeline:
        return {"status": "ok", "message": "Welcome to the Educational AI System API!"}
    return {"status": "error", "message": "RAG Pipeline is not initialized."}


@app.post("/generate-question", summary="새로운 문제 생성", response_model=List[ProblemResponse])
async def generate_question_endpoint(request: QuestionRequest) -> List[Dict[str, Any]]:
    if not pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline is not available.")
    try:
        questions = pipeline.generate_questions(
            subject=request.subject,
            unit=request.unit,
            difficulty=request.difficulty,
            count=request.count,
        )
        
        # DB 테이블 형식에 맞춰 응답 변환
        db_formatted_questions = []
        for q in questions:
            # 필요한 필드만 선택하여 DB 형식으로 변환
            db_question = {
                'id': q.get('id'),
                'title': q.get('title'),
                'description': q.get('description'),
                'content': q.get('content'),
                'type': q.get('type'),
                'difficulty': q.get('difficulty'),
                'subject': q.get('subject'),
                'gradeLevel': q.get('gradeLevel'),
                'unit': q.get('unit'),
                'options': q.get('options'),
                'correctAnswer': q.get('correctAnswer'),
                'explanation': q.get('explanation'),
                'hints': q.get('hints'),
                'tags': q.get('tags'),
                'points': q.get('points', 1),
                'timeLimit': q.get('timeLimit'),
                'isActive': q.get('isActive', True),
                'isAIGenerated': q.get('isAIGenerated', True),
                'aiGenerationId': q.get('aiGenerationId'),
                'qualityScore': q.get('qualityScore'),
                'reviewStatus': q.get('reviewStatus', 'PENDING'),
                'status': q.get('status', 'DRAFT'),
                'modelName': q.get('modelName'),
                'tokensUsed': q.get('tokensUsed'),
                'costUsd': q.get('costUsd'),
                'createdAt': q.get('createdAt'),
                'updatedAt': q.get('updatedAt')
            }
            db_formatted_questions.append(db_question)
        
        return db_formatted_questions
    except Exception as e:
        logger.error(f"An unexpected error occurred during question generation: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")

@app.post("/generate-report", summary="학생 학습 리포트 생성", response_model=ReportResponse)
async def generate_report_endpoint(request: ReportRequest) -> Dict[str, Any]:
    """
    학생의 학습 데이터를 분석하여 주간 학습 리포트를 생성합니다.
    
    - **user_id**: 분석할 학생의 ID
    
    Returns:
    - 학생의 학습 분석 리포트 (강점, 약점, 권장사항 포함)
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline is not available.")
    
    try:
        logger.info(f"Generating report for user: {request.user_id}")
        
        # LLM 클라이언트와 StudentAnalyzer 초기화
        llm_client = pipeline.llm_client
        analyzer = StudentAnalyzer(llm_client)
        
        # 리포트 생성
        report_data = analyzer.analyze(request.user_id)
        
        # 에러 체크
        if "error" in report_data:
            raise HTTPException(status_code=404, detail=report_data["error"])
        
        # 응답 형식 구성
        from datetime import datetime
        response = {
            "user_id": request.user_id,
            "report_text": report_data.get("report_text", ""),
            "weakest_unit": report_data.get("analysis_data", {}).get("weakest_unit", ""),
            "performance_summary": report_data.get("analysis_data", {}).get("performance_summary", {}),
            "generated_at": datetime.now().isoformat()
        }
        
        logger.info(f"Report generated successfully for user: {request.user_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.post("/chat/message", summary="챗봇과 메시지 주고받기 (REST)")
async def chat_message_endpoint(request: ChatRequest) -> ChatResponse:
    if not pipeline:
        raise HTTPException(status_code=503, detail="Core services are not available.")
    
    try:
        tutor = ChatbotTutor(request.user_id, pipeline.llm_client)
        if not tutor.weekly_report_context:
            raise HTTPException(status_code=404, detail=f"No report found for user {request.user_id}.")

        # 서버에 저장된 히스토리 가져오기 또는 초기화
        history = chat_histories.get(request.user_id, [])
        if not history:
            _, initial_history = tutor.start_session()
            history.extend(initial_history)

        history.append({"role": "user", "content": request.user_message})
        ai_response = tutor.get_response(request.user_message, history)
        history.append({"role": "assistant", "content": ai_response})
        
        # 업데이트된 히스토리 저장
        chat_histories[request.user_id] = history
        
        return ChatResponse(ai_response=ai_response)

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
        history = chat_histories.get(user_id)

        # 세션 시작 및 첫 메시지 전송
        if not history:
            initial_message, new_history = tutor.start_session()
            if not tutor.weekly_report_context:
                 await websocket.send_text(initial_message) # 리포트 없다는 메시지
                 await websocket.close()
                 return
            history = new_history
            chat_histories[user_id] = history
            await websocket.send_text(initial_message)
        else:
            await websocket.send_text("대화를 다시 시작합니다. 마지막으로 어떤 이야기를 했었죠?")

        # 대화 루프
        while True:
            user_message = await websocket.receive_text()
            history.append({"role": "user", "content": user_message})
            
            ai_response = tutor.get_response(user_message, history)
            history.append({"role": "assistant", "content": ai_response})
            
            chat_histories[user_id] = history # 매번 히스토리 업데이트
            await websocket.send_text(ai_response)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user: {user_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket chat for user {user_id}: {e}")
        await websocket.close(code=1011, reason="An internal error occurred.")

# 서버 실행을 위한 uvicorn 명령어 (터미널에서 실행):
# uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
