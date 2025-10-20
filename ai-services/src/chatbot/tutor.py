from typing import Dict, Any, List, Tuple
import json

from ..models.llm_client import LLMClient
from ..utils.db import get_db_connection
from ..utils.logger import get_logger

class ChatbotTutor:
    """
    학생의 주간 리포트를 기반으로 대화형 학습을 제공하는 챗봇입니다.
    """

    def __init__(self, user_id: str, llm_client: LLMClient):
        """
        ChatbotTutor를 초기화합니다.
        """
        self.user_id = user_id
        self.llm_client = llm_client
        self.logger = get_logger(__name__)
        self.analysis_context = None
        self._load_latest_report_context()

    def _load_latest_report_context(self):
        """가장 최근의 주간 리포트를 DB에서 로드하여 대화의 컨텍스트를 설정합니다."""
        self.logger.info(f"Loading latest report context for user: {self.user_id}")
        query = """
        SELECT content, analysisData 
        FROM teacher_reports 
        WHERE json_extract(students, '$[0].id') = ?
        ORDER BY createdAt DESC LIMIT 1;
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                # In SQLite, we use `?` for parameters
                cursor.execute(query, (self.user_id,))
                report = cursor.fetchone()
            
            if report:
                report_text, analysis_data_json = report
                analysis_data = json.loads(analysis_data_json)
                self.analysis_context = {
                    "report_text": report_text,
                    "analysis_data": analysis_data
                }
                self.logger.info(f"Successfully loaded report context for user {self.user_id}")
            else:
                self.logger.warning(f"No report found for user {self.user_id}")
                self.analysis_context = None

        except Exception as e:
            self.logger.error(f"An exception occurred while loading report context: {e}")
            self.analysis_context = None

    def start_session(self) -> Tuple[str, List[Dict[str, str]]]:
        """
        챗봇 세션을 시작하고, 첫 번째 인사 메시지를 생성합니다.

        Returns:
            (인사 메시지, 빈 대화 기록)
        """
        if not self.analysis_context:
            return "죄송합니다, 학생 데이터를 불러오는 데 실패했습니다. 먼저 학습 기록을 만들어주세요.", []

        weakest_unit = self.analysis_context.get("analysis_data", {}).get("weakest_unit", "알 수 없는 단원")

        initial_message = (
            f"안녕하세요, {self.user_id}님! AI 튜터입니다.\n"
            f"학습 기록을 분석해보니, '{weakest_unit}' 단원에서 어려움을 겪고 계신 것 같아요.\n"
            f"이 부분에 대해 함께 공부해볼까요? 원하신다면 개념 설명부터 시작하거나, 관련 문제를 풀어볼 수 있습니다."
        )
        
        history = [
            {"role": "assistant", "content": initial_message}
        ]
        
        return initial_message, history

    def get_response(self, user_message: str, history: List[Dict[str, str]]) -> str:
        """
        사용자 메시지와 대화 기록을 바탕으로 LLM의 답변을 생성합니다.

        Args:
            user_message: 사용자의 최신 메시지.
            history: 현재까지의 대화 기록.

        Returns:
            챗봇의 답변 메시지.
        """
        if not self.analysis_context:
            return "죄송합니다, 학생 데이터가 로드되지 않아 답변할 수 없습니다."

        weakest_unit = self.analysis_context.get("analysis_data", {}).get("weakest_unit")
        performance_summary = self.analysis_context.get("analysis_data", {}).get("performance_summary")

        prompt = f"""
        You are a friendly and patient AI tutor. Your goal is to help a student understand their weak subject area.

        **Student's Context:**
        - Weakest Unit: {weakest_unit}
        - Performance Summary: {performance_summary}

        **Your Task:**
        - Have a conversation with the student to help them learn the material.
        - You can both proactively teach concepts and answer the student's questions.
        - Guide the student with Socratic questioning. Do not give away direct answers to problems immediately. Provide hints first.
        - Keep your responses concise and easy to understand.

        **Conversation History:**
        {history}

        **Student's Latest Message:**
        {user_message}

        Generate the next response in the conversation.
        """

        try:
            ai_response = self.llm_client.generate_response(prompt)
            return ai_response
        except Exception as e:
            self.logger.error(f"Error generating chatbot response: {e}")
            return "죄송합니다, 답변을 생성하는 중 오류가 발생했습니다."
