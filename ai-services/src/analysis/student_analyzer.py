import pandas as pd
from typing import Dict, Any
from sqlalchemy import text

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

    def _fetch_logs(self, user_id: str) -> pd.DataFrame:
        """
        특정 사용자의 문제 풀이 로그를 데이터베이스에서 가져옵니다.

        Args:
            user_id: 분석할 학생의 ID.

        Returns:
            로그 데이터가 담긴 pandas DataFrame.
        """
        self.logger.info(f"Fetching logs for user: {user_id}")

        query = """
        SELECT
            a."isCorrect" AS isCorrect,
            a."timeSpent" AS timeSpent,
            p.subject,
            p.unit,
            p.difficulty
        FROM attempts a
        JOIN problems p ON a."problemId" = p.id
        WHERE a."userId" = :user_id;
        """
        
        try:
            with get_db_connection() as conn:
                df = pd.read_sql(text(query), conn, params={"user_id": user_id})

            # 드라이버/스키마 차이를 흡수하기 위해 컬럼명을 표준 형태로 정규화
            df = self._normalize_logs_df(df)
            if df.empty:
                return df

            self.logger.info(f"Fetched {len(df)} logs for user {user_id}.")
            return df
        except Exception as e:
            self.logger.error(f"로그 조회 중 오류: {e}")
            return pd.DataFrame()

    def _normalize_logs_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """필수 컬럼명이 존재하도록 정규화합니다."""
        if df.empty:
            return df

        normalized_cols = {c.lower(): c for c in df.columns}
        rename_map: Dict[str, str] = {}

        if 'iscorrect' in normalized_cols:
            rename_map[normalized_cols['iscorrect']] = 'isCorrect'
        if 'is_correct' in normalized_cols:
            rename_map[normalized_cols['is_correct']] = 'isCorrect'
        if 'timespent' in normalized_cols:
            rename_map[normalized_cols['timespent']] = 'timeSpent'
        if 'time_spent' in normalized_cols:
            rename_map[normalized_cols['time_spent']] = 'timeSpent'

        df = df.rename(columns=rename_map)

        # 분석에 필요한 최소 컬럼 집합
        required = {"isCorrect", "timeSpent", "subject", "unit", "difficulty"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            self.logger.error(f"로그에서 필요한 컬럼 누락: {missing}")
            return pd.DataFrame()

        return df

    def _summarize_logs(self, logs_df: pd.DataFrame) -> Dict[str, Any]:
        """
        로그 데이터를 분석하여 통계를 요약합니다.

        Args:
            logs_df: 로그 데이터프레임.

        Returns:
            분석된 통계가 담긴 딕셔너리.
        """
        if logs_df.empty:
            return {"error": "No logs found for this user."}

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
        
        self.logger.info("사용자 로그 요약을 생성했습니다.")
        return summary

    def _generate_prompt(self, summary: Dict[str, Any]) -> str:
        """
        LLM에 보낼 프롬프트를 생성합니다. 출력은 JSON 형식이어야 합니다.
        """
        prompt = f"""
        You are an expert educational consultant. Analyze the student's learning data below and create a comprehensive learning report.
        Your output MUST be a single valid JSON object with two keys: "weakest_unit" and "report_text".
        - "weakest_unit": A string containing the name of the single unit the student is weakest in.
        - "report_text": A string containing the full, human-readable report in Korean.

        **Learning Data Summary:**
        - Total problems solved: {summary.get('total_problems_solved')}
        - Overall correct rate: {summary.get('overall_correct_rate')}
        - Average time spent per problem: {summary.get('average_time_spent_seconds')} seconds

        **Performance by Subject:**
        {summary.get('performance_by_subject')}

        **Performance by Unit:**
        {summary.get('performance_by_unit')}

        **Performance by Difficulty:**
        {summary.get('performance_by_difficulty')}

        **Report Generation Guide (for the "report_text" field):**
        1.  **Overall Assessment**: Write a general evaluation of the student's current learning status.
        2.  **Strengths**: Based on the data, praise the student for subjects or units where they show strength.
        3.  **Weaknesses**: Based on the data, point out which subjects or units need improvement. Focus on the units with the lowest correct rate.
        4.  **Recommendations**: Recommend specific learning strategies or additional materials to address weaknesses and maintain strengths.

        Based on the guide above, write a kind and detailed report in the "report_text" field so the student can clearly understand their status and plan their next steps.
        Remember to identify the single weakest unit for the "weakest_unit" field.
        """
        return prompt

    def analyze(self, user_id: str) -> Dict[str, Any]:
        """
        학생의 학습 로그를 분석하여 최종 리포트를 JSON 형식으로 생성합니다.
        """
        logs_df = self._fetch_logs(user_id)
        if logs_df.empty:
            return {"error": "해당 사용자에 대한 학습 로그를 찾을 수 없습니다."}

        summary = self._summarize_logs(logs_df)
        if "error" in summary:
            return summary

        prompt = self._generate_prompt(summary)

        self.logger.info(f"사용자 {user_id}에 대한 분석 리포트를 생성합니다...")

        try:
            # LLMClient를 사용하여 구조화된 리포트를 생성
            structured_report = self.llm_client.generate_structured_response(prompt, response_format="json")
            
            # 최종 결과 조합
            final_output = {
                "report_text": structured_report.get("report_text", "리포트 텍스트를 생성하지 못했습니다."),
                "analysis_data": {
                    "weakest_unit": structured_report.get("weakest_unit", "취약 단원을 식별하지 못했습니다."),
                    "performance_summary": summary
                }
            }
            
            self.logger.info(f"사용자 {user_id} 리포트 생성 완료")
            return final_output
        except Exception as e:
            self.logger.error(f"리포트 생성 중 오류: {e}")
            return {{"error": "리포트 생성 중 오류가 발생했습니다."}}
