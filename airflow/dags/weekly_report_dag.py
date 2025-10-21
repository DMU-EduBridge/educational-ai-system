
from __future__ import annotations

import pendulum
import logging
import sys
import os
import json
import uuid

from airflow.models.dag import DAG
from airflow.providers.standard.operators.python import PythonOperator
from sqlalchemy import text

# 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ai_services_path = os.path.join(project_root, 'ai-services')
sys.path.insert(0, ai_services_path)

from src.analysis.student_analyzer import StudentAnalyzer
from src.models.llm_client import LLMClient
from src.utils.db import get_db_connection

# Configure logging
logger = logging.getLogger(__name__)

def get_all_user_ids():
    """DB에서 모든 학생 사용자 ID를 가져옵니다."""
    logger.info("Fetching all student user IDs from the database.")
    # 참고: 실제 사용자 테이블과 역할(예: 'student')을 확인해야 합니다.
    try:
        with get_db_connection() as conn:
            # 이 쿼리는 'users' 테이블에 'role' 컬럼이 있다고 가정합니다.
            # 실제 스키마에 맞게 수정이 필요할 수 있습니다.
            result = conn.execute(text("SELECT id FROM users WHERE role = 'STUDENT';")).fetchall()
            user_ids = [row[0] for row in result]
            if not user_ids:
                logger.warning("분석할 학생 사용자를 찾을 수 없습니다.")
                return []
            logger.info(f"Found {len(user_ids)} student users.")
            return user_ids
    except Exception as e:
        logger.error(f"사용자 ID 조회 중 오류 발생: {e}")
        return [] # 오류 발생 시 빈 리스트 반환

def generate_and_save_report(user_id: str, **kwargs):
    """
    리포트를 생성하고 teacher_reports 테이블에 저장합니다.
    """
    logger.info(f"리포트 생성 및 저장 시작: 사용자 ID {user_id}")
    try:
        llm_client = LLMClient()
        analyzer = StudentAnalyzer(llm_client)
        report_data = analyzer.analyze(user_id)

        if "error" in report_data:
            logger.error(f"리포트 생성 실패: {report_data['error']}")
            return

        report_id = str(uuid.uuid4())
        title = f"주간 학습 리포트 - {user_id}"
        content = report_data.get("report_text", "리포트 내용을 생성하지 못했습니다.")
        analysis_data = report_data.get("analysis_data", {})

        with get_db_connection() as conn:
            stmt = text("""
                INSERT INTO teacher_reports 
                (id, title, content, "analysisData", "createdBy", students, "reportType", status)
                VALUES (:id, :title, :content, :analysisData, :createdBy, :students, 'PROGRESS_REPORT', 'DRAFT')
            """)
            conn.execute(stmt, {
                "id": report_id,
                "title": title,
                "content": content,
                "analysisData": json.dumps(analysis_data, ensure_ascii=False),
                "createdBy": "airflow",
                "students": json.dumps([{"id": user_id}]),
            })
            # conn.commit() # autocommit 모드인 경우 필요 없음

        logger.info(f"리포트 저장 완료: 사용자 ID {user_id}, 리포트 ID {report_id}")

    except Exception as e:
        logger.error(f"리포트 생성 및 저장 중 오류 발생: {e}", exc_info=True)
        raise

with DAG(
    dag_id="weekly_learning_report",
    start_date=pendulum.datetime(2023, 10, 26, tz="Asia/Seoul"),
    schedule="0 0 * * 1",  # 매주 월요일 00:00에 실행
    catchup=False,
    tags=["reporting", "analysis"],
    doc_md="""
    ### 주간 학습 리포트 생성 DAG

    이 DAG는 매주 학생들의 학습 데이터를 분석하여 주간 리포트를 생성하고,
    `teacher_reports` 테이블에 저장하는 역할을 합니다.
    """,
) as dag:
    users_to_process = get_all_user_ids()

    if users_to_process:
        for user_id in users_to_process:
            PythonOperator(
                task_id=f"generate_and_save_report_for_{user_id}",
                python_callable=generate_and_save_report,
                op_kwargs={"user_id": user_id},
            )

    else:
        logger.warning("리포트를 생성할 사용자가 없습니다.")
