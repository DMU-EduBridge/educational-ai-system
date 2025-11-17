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
        self._load_or_generate_context()

    def _load_or_generate_context(self):
        """
        DB에서 주간 리포트를 로드하거나, 없으면 실시간으로 생성합니다.
        """
        self.logger.info(f"Loading or generating context for user: {self.user_id}")
        
        # 1단계: DB에서 기존 리포트 찾기 시도
        from sqlalchemy import text
        from ..utils.config import get_settings
        
        settings = get_settings()
        
        # 데이터베이스 타입에 따라 JSON 쿼리 구문 변경
        if settings.database_url and settings.database_url.startswith("postgresql"):
            # PostgreSQL: JSON 연산자 사용 (컬럼명 대소문자 구분)
            query = text("""
            SELECT content, "analysisData" 
            FROM teacher_reports 
            WHERE students::jsonb @> :students_filter
            ORDER BY "createdAt" DESC LIMIT 1
            """)
            params = {"students_filter": json.dumps([{"id": self.user_id}])}
        else:
            # SQLite: json_extract 사용
            query = text("""
            SELECT content, analysisData 
            FROM teacher_reports 
            WHERE json_extract(students, '$[0].id') = :user_id
            ORDER BY createdAt DESC LIMIT 1
            """)
            params = {"user_id": self.user_id}
        
        try:
            with get_db_connection() as conn:
                result = conn.execute(query, params)
                report = result.fetchone()
            
            if report:
                report_text, analysis_data_json = report
                self.weekly_report_context = {
                    "report_text": report_text,
                    "analysis_data": json.loads(analysis_data_json)
                }
                self.logger.info(f"Successfully loaded weekly report from DB for user {self.user_id}")
                return
        except Exception as e:
            self.logger.warning(f"Could not load weekly report from DB: {e}")
        
        # 2단계: DB에 리포트가 없으면 실시간으로 분석 수행
        self.logger.info(f"No DB report found. Generating real-time analysis for user {self.user_id}")
        try:
            analysis_result = self.student_analyzer.analyze(self.user_id)
            
            if "error" not in analysis_result:
                self.weekly_report_context = {
                    "report_text": analysis_result.get("report_text", ""),
                    "analysis_data": analysis_result.get("analysis_data", {})
                }
                self.logger.info(f"Successfully generated real-time analysis for user {self.user_id}")
            else:
                self.logger.warning(f"Could not generate analysis: {analysis_result.get('error')}")
                self.weekly_report_context = None
        except Exception as e:
            self.logger.error(f"Error generating real-time analysis: {e}")
            self.weekly_report_context = None

    def _load_weekly_report_context(self):
        """
        [Deprecated] DB에서 주간 리포트를 로드합니다.
        이 메서드는 _load_or_generate_context()로 대체되었습니다.
        """
        self._load_or_generate_context()

    def _get_real_time_analysis(self) -> Dict[str, Any]:
        """최근 1일간의 로그를 실시간으로 분석합니다."""
        self.logger.info(f"Performing real-time analysis for user: {self.user_id}")
        return self.student_analyzer.analyze(self.user_id, time_window_days=1)

    def start_session(self) -> Tuple[str, List[Dict[str, str]]]:
        """
        챗봇 세션을 시작하고, 첫 번째 인사 메시지를 생성합니다.
        """
        if not self.weekly_report_context:
            # 리포트 생성 실패 시 기본 메시지
            return (
                f"안녕하세요! AI 튜터입니다.\n"
                "아직 분석할 학습 데이터가 충분하지 않네요. "
                "문제를 풀고 나면 학습 분석을 통해 더 도움을 드릴 수 있어요. "
                "궁금한 점이 있으면 편하게 질문해주세요!"
            ), []

        weakest_unit = self.weekly_report_context.get("analysis_data", {}).get("weakest_unit", "특정 단원")
        performance_summary = self.weekly_report_context.get("analysis_data", {}).get("performance_summary", {})
        
        # 전체 정답률 계산
        total_correct = performance_summary.get("total_correct", 0)
        total_attempted = performance_summary.get("total_attempted", 0)
        accuracy = (total_correct / total_attempted * 100) if total_attempted > 0 else 0

        initial_message = (
            f"안녕하세요! AI 튜터입니다.\n\n"
            f"학습 분석 결과를 확인해보니:\n"
            f"📊 총 {total_attempted}문제를 풀었고, 정답률은 {accuracy:.1f}%입니다.\n"
        )
        
        if weakest_unit and weakest_unit != "특정 단원":
            initial_message += f"🎯 '{weakest_unit}' 단원에서 개선이 필요해 보여요.\n\n"
        
        initial_message += "이 부분에 대해 함께 공부해볼까요? 또는, 다른 궁금한 점이 있다면 편하게 질문해주세요."
        
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
        당신은 친절하고 인내심 있는 AI 튜터입니다. 당신의 목표는 학생의 학습 성과 데이터를 기반으로 학습을 돕는 것입니다.

        **학생의 학습 맥락:**
        {base_context_str}
        {real_time_context_str}

        **당신의 임무:**
        - 학생과 대화하며 학습을 도와주세요.
        - 사용자가 최근 성과에 대해 물으면, 실시간 데이터를 사용하여 답변해주세요.
        - 그 외의 경우에는 주간 리포트에서 파악된 취약점에 초점을 맞춰 대화하세요.
        - 소크라테스식 질문법을 사용하여 학생을 지도하고, 문제에 대한 정답을 즉시 알려주지 마세요. 먼저 힌트를 제공하세요.

        **답변 근거에 대한 안내:**
        - 사용자가 답변의 근거나 출처를 물어보면, 다음과 같이 설명해주세요:
          "제 답변은 세 가지 주요 데이터를 기반으로 합니다:
          1) 학생님의 문제 풀이 학습 로그 데이터
          2) 가장 최신의 주간/실시간 학습 분석 리포트
          3) 대한민국 정규 교육과정 중학교 3학년 천재교육 수학 교과서
          이러한 자료들을 종합하여 학생님께 맞춤형 학습 지도를 제공하고 있습니다."

        **대화 기록:**
        {history}

        **학생의 최근 메시지:**
        {user_message}

        이제 대화의 다음 응답을 한국어로 생성해주세요.
        """

        try:
            ai_response = self.llm_client.generate_response(prompt)
            return ai_response
        except Exception as e:
            self.logger.error(f"Error generating chatbot response: {e}")
            return "죄송합니다, 답변을 생성하는 중 오류가 발생했습니다."
