import sys
from pathlib import Path
from datetime import datetime, timedelta

from airflow.decorators import dag, task

# 프로젝트의 루트 디렉토리를 Python 경로에 추가
# 이렇게 해야 Airflow가 ai-services 모듈을 찾을 수 있습니다.
project_root = Path('/opt/airflow/project')
ai_services_dir = project_root / 'ai-services'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(ai_services_dir))

from ai_services.src.utils.db import get_db_connection
from ai_services.src.analysis.student_analyzer import StudentAnalyzer
from ai_services.src.utils.report_writer import save_report_to_db
from ai_services.src.models.llm_client import LLMClient
from ai_services.src.utils.config import get_settings

@dag(
    dag_id='weekly_student_reports',
    schedule_interval='@weekly',
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['reporting'],
    doc_md="""
    ### Weekly Student Reports DAG

    This DAG generates a weekly performance report for all students who were active in the last 7 days.
    - It fetches the list of active students.
    - For each student, it generates a report using the StudentAnalyzer.
    - It saves the generated report to the `teacher_reports` table in the database.
    """
)
def weekly_reports_dag():
    """
    주간 학생 리포트 생성 DAG
    """

    @task
    def get_active_students() -> list[str]:
        """지난 7일간 활동한 학생들의 ID 목록을 가져옵니다."""
        print("Fetching active students from the last 7 days...")
        
        query = "SELECT DISTINCT userId FROM attempts WHERE updatedAt >= date('now', '-7 days');"
        
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                students = [row[0] for row in cursor.fetchall()]
            print(f"Found {len(students)} active students.")
            return students
        except Exception as e:
            print(f"Error fetching active students: {e}")
            return []

    @task
    def generate_and_save_report(user_id: str):
        """특정 학생의 리포트를 생성하고 데이터베이스에 저장합니다."""
        print(f"Generating report for student: {user_id}")
        
        try:
            # LLM 클라이언트 및 분석기 초기화
            # Airflow 환경에서는 전역 pipeline 객체를 사용할 수 없으므로 직접 생성합니다.
            settings = get_settings()
            llm_client = LLMClient(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key
            )
            analyzer = StudentAnalyzer(llm_client)
            
            # 리포트 생성
            report_data = analyzer.analyze(user_id)
            
            if "error" in report_data:
                print(f"Skipping report for {user_id} due to analysis error: {report_data['error']}")
                return

            # DB에 저장
            save_report_to_db(user_id, report_data)
            print(f"Successfully generated and saved report for student: {user_id}")

        except Exception as e:
            print(f"Failed to generate or save report for student {user_id}: {e}")
            # Airflow에서 이 task를 실패로 표시하기 위해 예외를 다시 발생시킬 수 있습니다.
            raise

    # 작업 흐름 정의
    active_student_ids = get_active_students()
    generate_and_save_report.expand(user_id=active_student_ids)

# DAG 인스턴스화
weekly_reports_dag_instance = weekly_reports_dag()
