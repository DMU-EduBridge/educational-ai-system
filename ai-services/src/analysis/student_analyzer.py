import pandas as pd
from typing import List, Dict, Any

from ..utils.db import get_db_connection
from ..models.llm_client import LLMClient
from ..utils.logger import get_logger

class StudentAnalyzer:
    """학생의 문제 풀이 로그를 분석하고 종합 리포트를 생성합니다."""

    def __init__(self, llm_client: LLMClient):
        """
        StudentAnalyzer를 초기화합니다.

        Args:
            llm_client: LLM 클라이언트 인스턴스.
        """
        self.llm_client = llm_client
        self.logger = get_logger(__name__)

    def _fetch_logs(self, user_id: str, time_window_days: int = None) -> pd.DataFrame:
        """
        특정 사용자의 문제 풀이 로그를 데이터베이스에서 가져옵니다.

        Args:
            user_id: 분석할 학생의 ID.
            time_window_days: 로그를 가져올 최근 기간(일). None이면 전체 기간.

        Returns:
            로그 데이터가 담긴 pandas DataFrame.
        """
        self.logger.info(f"Fetching logs for user: {user_id} (last {time_window_days or 'all'} days)")
        
        query = """
        SELECT
            pl.isCorrect,
            pl.timeSpent,
            p.subject,
            p.unit,
            p.difficulty
        FROM attempts pl
        JOIN problems p ON pl.problemId = p.id
        WHERE pl.userId = ?
        """
        params = [user_id]

        if time_window_days:
            query += " AND pl.updatedAt >= date('now', ?);"
            params.append(f'-{time_window_days} days')

        try:
            with get_db_connection() as conn:
                df = pd.read_sql(query, conn, params=params)
            self.logger.info(f"Fetched {len(df)} logs for user {user_id}.")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching logs: {e}")
            return pd.DataFrame()

    def _summarize_logs(self, logs_df: pd.DataFrame) -> Dict[str, Any]:
        """
        로그 데이터를 분석하여 통계를 요약합니다.

        Args:
            logs_df: 로그 데이터프레임.

        Returns:
            분석된 통계가 담긴 딕셔너리.
        """
        if logs_df.empty:
            return {"error": "No logs found for this user for the given time period."}

        total_problems = len(logs_df)
        correct_answers = logs_df['isCorrect'].sum()
        overall_correct_rate = (correct_answers / total_problems) * 100 if total_problems > 0 else 0
        avg_time_spent = logs_df['timeSpent'].mean()

        summary = {
            "total_problems_solved": total_problems,
            "overall_correct_rate": f"{overall_correct_rate:.2f}%",
            "average_time_spent_seconds": f"{avg_time_spent:.2f}",
            "performance_by_subject": logs_df.groupby('subject')['isCorrect'].value_counts(normalize=True).unstack().fillna(0).to_dict(),
            "performance_by_unit": logs_df.groupby(['subject', 'unit'])['isCorrect'].value_counts(normalize=True).unstack().fillna(0).to_dict(),
            "performance_by_difficulty": logs_df.groupby('difficulty')['isCorrect'].value_counts(normalize=True).unstack().fillna(0).to_dict(),
        }
        
        self.logger.info(f"Log summary created for user.")
        return summary

    def _generate_prompt(self, summary: Dict[str, Any], is_real_time: bool = False) -> str:
        """
        LLM에 보낼 프롬프트를 생성합니다. JSON 출력을 요청합니다.
        """
        if is_real_time:
            report_intro = "Analyze the student's RECENT learning data below and create a brief, real-time analysis."
            report_guide = "Briefly summarize the student's performance on the problems they solved recently."
        else:
            report_intro = "Analyze the student's learning data below and create a comprehensive learning report."
            report_guide = """
            1.  **Overall Assessment**: Write a general evaluation of the student's current learning status.
            2.  **Strengths**: Based on the data, praise the student for subjects or units where they show strength.
            3.  **Weaknesses**: Based on the data, point out which subjects or units need improvement. Focus on the units with the lowest correct rate.
            4.  **Recommendations**: Recommend specific learning strategies or additional materials to address weaknesses and maintain strengths.
            """

        prompt = f"""
        You are an expert educational consultant. {report_intro}
        Your output MUST be a single valid JSON object with two keys: "weakest_unit" and "report_text".
        - "weakest_unit": A string containing the name of the single unit the student is weakest in. If there are no clear weaknesses, this can be null.
        - "report_text": A string containing the analysis report in Korean.

        **Learning Data Summary:**
        {summary}

        **Report Generation Guide (for the "report_text" field):**
        {report_guide}

        Based on the guide above, write a kind and detailed report in the "report_text" field.
        """
        return prompt

    def analyze(self, user_id: str, time_window_days: int = None) -> Dict[str, Any]:
        """
        학생의 학습 로그를 분석하여 최종 리포트를 JSON 형식으로 생성합니다.
        """
        logs_df = self._fetch_logs(user_id, time_window_days)
        if logs_df.empty:
            return {"error": "해당 기간 동안 사용자의 학습 로그를 찾을 수 없습니다."}

        summary = self._summarize_logs(logs_df)
        if "error" in summary:
            return summary

        is_real_time = time_window_days is not None
        prompt = self._generate_prompt(summary, is_real_time=is_real_time)

        self.logger.info(f"Generating analysis report for user {user_id}...")

        try:
            structured_report = self.llm_client.generate_structured_response(prompt, response_format="json")
            
            final_output = {
                "report_text": structured_report.get("report_text", "리포트 텍스트를 생성하지 못했습니다."),
                "analysis_data": {
                    "weakest_unit": structured_report.get("weakest_unit"),
                    "performance_summary": summary
                }
            }
            
            self.logger.info(f"Successfully generated structured report for user {user_id}.")
            return final_output
        except Exception as e:
            self.logger.error(f"Error generating structured report: {e}")
            return {{"error": "리포트 생성 중 오류가 발생했습니다."}}
