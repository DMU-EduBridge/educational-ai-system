import json
import uuid
from datetime import datetime
from sqlalchemy import text

from .db import get_db_connection
from ..utils.logger import get_logger

def save_report_to_db(user_id: str, report_data: dict):
    """
    생성된 분석 리포트를 데이터베이스의 teacher_reports 테이블에 저장합니다.

    Args:
        user_id (str): 학생의 ID.
        report_data (dict): StudentAnalyzer.analyze()가 반환한 딕셔너리.
    """
    logger = get_logger(__name__)
    logger.info(f"Saving report for user {user_id} to the database.")

    report_id = str(uuid.uuid4())
    title = f"주간 학습 리포트 (학생: {user_id})"
    content = report_data.get("report_text", "")
    analysis_data_json = json.dumps(report_data.get("analysis_data", {}), ensure_ascii=False)
    students_json = json.dumps([{"id": user_id}])
    created_by = "api"
    now = datetime.utcnow()

    # SQLAlchemy text() 사용 및 named parameter 방식
    query = text("""
    INSERT INTO teacher_reports (
        id, title, content, reportType, students, analysisData, createdBy, createdAt, updatedAt, status
    ) VALUES (:id, :title, :content, :reportType, :students, :analysisData, :createdBy, :createdAt, :updatedAt, :status)
    """)
    
    params = {
        "id": report_id,
        "title": title,
        "content": content,
        "reportType": 'PROGRESS_REPORT',
        "students": students_json,
        "analysisData": analysis_data_json,
        "createdBy": created_by,
        "createdAt": now,
        "updatedAt": now,
        "status": 'DRAFT'
    }

    try:
        with get_db_connection() as conn:
            conn.execute(query, params)
            conn.commit()
        logger.info(f"Successfully saved report {report_id} for user {user_id}.")
    except Exception as e:
        logger.error(f"Failed to save report for user {user_id}: {e}")
        raise
