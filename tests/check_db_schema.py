#!/usr/bin/env python3
"""
DB 스키마 확인 스크립트
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent / "ai-services"
sys.path.insert(0, str(project_root))

from src.utils.db import get_db_connection
from sqlalchemy import text

def check_db_schema():
    """DB 스키마와 데이터 확인"""
    
    print("=" * 60)
    print("🔍 데이터베이스 스키마 확인")
    print("=" * 60)
    print()
    
    try:
        with get_db_connection() as conn:
            # 테이블 목록 확인
            print("📋 테이블 목록:")
            tables_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name;
            """
            tables = conn.execute(text(tables_query)).fetchall()
            for table in tables:
                print(f"  - {table[0]}")
            print()
            
            # users 테이블 확인
            if any('users' in str(t[0]).lower() for t in tables):
                print("👤 Users 테이블:")
                users_count = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
                print(f"  총 사용자 수: {users_count}")
                
                if users_count > 0:
                    sample_users = conn.execute(
                        text("SELECT id, email, name, role FROM users LIMIT 3")
                    ).fetchall()
                    print("  샘플 사용자:")
                    for user in sample_users:
                        print(f"    - ID: {user[0]}, Email: {user[1]}, Name: {user[2]}, Role: {user[3]}")
                print()
            
            # problems 테이블 확인
            if any('problem' in str(t[0]).lower() for t in tables):
                problem_table = [t[0] for t in tables if 'problem' in str(t[0]).lower()][0]
                print(f"📝 {problem_table.capitalize()} 테이블:")
                problems_count = conn.execute(text(f"SELECT COUNT(*) FROM {problem_table}")).scalar()
                print(f"  총 문제 수: {problems_count}")
                
                if problems_count > 0:
                    sample_problems = conn.execute(
                        text(f"SELECT id, subject, unit, difficulty FROM {problem_table} LIMIT 3")
                    ).fetchall()
                    print("  샘플 문제:")
                    for prob in sample_problems:
                        print(f"    - ID: {prob[0]}, Subject: {prob[1]}, Unit: {prob[2]}, Difficulty: {prob[3]}")
                print()
            
            # attempts 테이블 확인
            if any('attempt' in str(t[0]).lower() for t in tables):
                attempt_table = [t[0] for t in tables if 'attempt' in str(t[0]).lower()][0]
                print(f"✏️  {attempt_table.capitalize()} 테이블:")
                attempts_count = conn.execute(text(f"SELECT COUNT(*) FROM {attempt_table}")).scalar()
                print(f"  총 시도 수: {attempts_count}")
                
                if attempts_count > 0:
                    # 컬럼명 확인
                    columns_query = f"""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = '{attempt_table}' 
                        ORDER BY ordinal_position;
                    """
                    columns = conn.execute(text(columns_query)).fetchall()
                    print(f"  컬럼: {', '.join([c[0] for c in columns])}")
                    
                    sample_attempts = conn.execute(
                        text(f"SELECT * FROM {attempt_table} LIMIT 3")
                    ).fetchall()
                    print(f"  샘플 데이터: {len(sample_attempts)}개")
                print()
            
            # teacher_reports 테이블 확인
            if any('report' in str(t[0]).lower() for t in tables):
                report_table = [t[0] for t in tables if 'report' in str(t[0]).lower()][0]
                print(f"📊 {report_table.capitalize()} 테이블:")
                reports_count = conn.execute(text(f"SELECT COUNT(*) FROM {report_table}")).scalar()
                print(f"  총 리포트 수: {reports_count}")
                print()
            
            print("=" * 60)
            print("✅ 스키마 확인 완료")
            print("=" * 60)
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_db_schema()
