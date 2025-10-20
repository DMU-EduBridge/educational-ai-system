from typing import Dict, Any, List, Tuple
import json

from ..models.llm_client import LLMClient
from ..analysis.student_analyzer import StudentAnalyzer
from ..utils.db import get_db_connection
from ..utils.logger import get_logger

class ChatbotTutor:
    """
    학생의 주간 리포트와 실시간 로그를 모두 활용하는 하이브리드 챗봇입니다.
    """

    def __init__(self, user_id: str, llm_client: LLMClient):
        """
        ChatbotTutor를 초기화합니다.
        """
        self.user_id = user_id
        self.llm_client = llm_client
        self.logger = get_logger(__name__)
        self.student_analyzer = StudentAnalyzer(llm_client)
        self.weekly_report_context = None
        self._load_weekly_report_context()

    def _load_weekly_report_context(self):
        """가장 최근의 주간 리포트를 DB에서 로드하여 대화의 기본 컨텍스트를 설정합니다."""
        self.logger.info(f"Loading weekly report context for user: {self.user_id}")
        query = """
        SELECT content, analysisData 
        FROM teacher_reports 
        WHERE json_extract(students, '$[0].id') = ?
        ORDER BY createdAt DESC LIMIT 1;
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (self.user_id,))
                report = cursor.fetchone()
            
            if report:
                report_text, analysis_data_json = report
                self.weekly_report_context = {
                    "report_text": report_text,
                    "analysis_data": json.loads(analysis_data_json)
                }
                self.logger.info(f"Successfully loaded weekly report for user {self.user_id}")
            else:
                self.logger.warning(f"No weekly report found for user {self.user_id}")
        except Exception as e:
            self.logger.error(f"An exception occurred while loading weekly report: {e}")

    def _get_real_time_analysis(self) -> Dict[str, Any]:
        """최근 1일간의 로그를 실시간으로 분석합니다."""
        self.logger.info(f"Performing real-time analysis for user: {self.user_id}")
        return self.student_analyzer.analyze(self.user_id, time_window_days=1)

    def start_session(self) -> Tuple[str, List[Dict[str, str]]]:
        """
        챗봇 세션을 시작하고, 첫 번째 인사 메시지를 생성합니다.
        """
        if not self.weekly_report_context:
            return "안녕하세요! 아직 분석된 주간 리포트가 없네요. 문제 풀이 기록이 쌓이면 제가 분석해서 알려드릴게요.", []

        weakest_unit = self.weekly_report_context.get("analysis_data", {}).get("weakest_unit", "- 아직 데이터가 부족해요 - ")

        initial_message = (
            f"안녕하세요, {self.user_id}님! AI 튜터입니다.\n"
            f"지난 주 학습 내용을 분석해보니, '{weakest_unit}' 단원에서 어려움을 겪고 계신 것 같아요.\n"
            f"이 부분에 대해 함께 공부해볼까요? 또는, 다른 궁금한 점이 있다면 편하게 질문해주세요."
        )
        
        history = [{"role": "assistant", "content": initial_message}]
        return initial_message, history

    def get_response(self, user_message: str, history: List[Dict[str, str]]) -> str:
        """
        사용자 메시지와 대화 기록을 바탕으로 LLM의 답변을 생성합니다.
        """
        real_time_keywords = ["오늘", "최근", "방금", "지금"]
        use_real_time_analysis = any(keyword in user_message for keyword in real_time_keywords)

        real_time_context_str = ""
        if use_real_time_analysis:
            real_time_analysis = self._get_real_time_analysis()
            if "error" not in real_time_analysis:
                real_time_context_str = f"\n**Student's Real-Time Performance (Today):**\n{real_time_analysis['analysis_data']['performance_summary']}"

        base_context_str = "No weekly report available."
        if self.weekly_report_context:
            base_context_str = f"""
            **Student's Weekly Report Summary:**
            - Weakest Unit: {self.weekly_report_context.get("analysis_data", {}).get("weakest_unit")}
            - Report: {self.weekly_report_context.get("report_text")} 
            """

        prompt = f"""
        You are a friendly and patient AI tutor. Your goal is to help a student learn based on their performance data.

        **Student's Context:**
        {base_context_str}
        {real_time_context_str}

        **Your Task:**
        - Have a conversation with the student to help them learn.
        - If the user asks about their recent performance, use the real-time data to answer.
        - Otherwise, focus on the weaknesses identified in the weekly report.
        - Guide the student with Socratic questioning. Do not give away direct answers to problems immediately. Provide hints first.

        **Conversation History:**
        {history}

        **Student's Latest Message:**
        {user_message}

        Generate the next response in the conversation in Korean.
        """

        try:
            ai_response = self.llm_client.generate_response(prompt)
            return ai_response
        except Exception as e:
            self.logger.error(f"Error generating chatbot response: {e}")
            return "죄송합니다, 답변을 생성하는 중 오류가 발생했습니다."
