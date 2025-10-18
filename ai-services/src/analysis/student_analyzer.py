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

    def _fetch_logs(self, user_id: str) -> pd.DataFrame:
        """
        특정 사용자의 문제 풀이 로그를 데이터베이스에서 가져옵니다.
        
        Args:
            user_id: 분석할 학생의 ID.

        Returns:
            로그 데이터가 담긴 pandas DataFrame.
        """
        self.logger.info(f"Fetching logs for user: {user_id}")
        
        # TODO: SQL 쿼리를 구체화해야 합니다.
        # problem_logs 테이블과 problems 테이블을 조인하여 과목, 단원, 난이도 정보를 함께 가져옵니다.
        query = """
        SELECT
            pl.isCorrect,
            pl.timeSpent,
            p.subject,
            p.unit,
            p.difficulty
        FROM problem_logs pl
        JOIN problems p ON pl.problemId = p.id
        WHERE pl.userId = %s;
        """
        
        try:
            with get_db_connection() as conn:
                df = pd.read_sql(query, conn, params=(user_id,))
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
        
        self.logger.info(f"Log summary created for user.")
        return summary

    def _generate_prompt(self, summary: Dict[str, Any]) -> str:
        """
        LLM에 보낼 프롬프트를 생성합니다.

        Args:
            summary: 요약된 통계 딕셔너리.

        Returns:
            생성된 프롬프트 문자열.
        """
        # TODO: 프롬프트를 더 정교하게 다듬어야 합니다.
        prompt = f"""
        당신은 전문 교육 컨설턴트입니다. 아래 학생의 학습 데이터를 분석하여 종합적인 학습 리포트를 작성해주세요.

        **학습 데이터 요약:**
        - 총 푼 문제 수: {summary.get('total_problems_solved')}
        - 전체 정답률: {summary.get('overall_correct_rate')}
        - 평균 문제 풀이 시간: {summary.get('average_time_spent_seconds')}초

        **과목별 분석:**
        {summary.get('performance_by_subject')}

        **단원별 분석:**
        {summary.get('performance_by_unit')}

        **난이도별 분석:**
        {summary.get('performance_by_difficulty')}

        **리포트 작성 가이드:**
        1.  **총평**: 학생의 현재 학습 상태에 대한 전반적인 평가를 작성해주세요.
        2.  **강점**: 어떤 과목이나 단원에서 강점을 보이는지 구체적인 데이터를 근거로 칭찬해주세요.
        3.  **약점**: 어떤 과목이나 단원에서 개선이 필요한지 구체적인 데이터를 근거로 지적해주세요. 특히 정답률이 낮은 단원을 중심으로 분석해주세요.
        4.  **학습 추천**: 약점을 보완하고 강점을 유지하기 위한 구체적인 학습 전략이나 추가 학습 자료를 추천해주세요.
        
        위 가이드에 따라, 학생이 자신의 학습 상태를 명확히 이해하고 다음 학습 계획을 세울 수 있도록 친절하고 상세하게 리포트를 작성해주세요.
        """
        return prompt

    def analyze(self, user_id: str) -> str:
        """
        학생의 학습 로그를 분석하여 최종 리포트를 생성합니다.

        Args:
            user_id: 분석할 학생의 ID.

        Returns:
            LLM이 생성한 분석 리포트.
        """
        logs_df = self._fetch_logs(user_id)
        if logs_df.empty:
            return "해당 사용자에 대한 학습 로그를 찾을 수 없습니다."
            
        summary = self._summarize_logs(logs_df)
        if "error" in summary:
            return summary["error"]
            
        prompt = self._generate_prompt(summary)
        
        self.logger.info(f"Generating analysis report for user {user_id}...")
        
        try:
            # LLMClient를 사용하여 리포트 생성
            report = self.llm_client.generate_response(prompt)
            self.logger.info(f"Successfully generated report for user {user_id}.")
            return report
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return "리포트 생성 중 오류가 발생했습니다."
