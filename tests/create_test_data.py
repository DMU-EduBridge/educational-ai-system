#!/usr/bin/env python3
"""
테스트용 학습 데이터 생성 스크립트
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent / "ai-services"
sys.path.insert(0, str(project_root))

from src.utils.db import get_db_connection
from sqlalchemy import text
import uuid
from datetime import datetime, timedelta
import random

def create_test_data():
    """테스트용 학생 데이터와 문제 풀이 로그 생성"""
    
    print("=" * 60)
    print("📝 테스트 데이터 생성 시작")
    print("=" * 60)
    
    try:
        with get_db_connection() as conn:
            # 1. 실제 학생 가져오기
            student = conn.execute(
                text("SELECT id, name FROM users WHERE role = 'STUDENT' LIMIT 1")
            ).fetchone()
            
            if not student:
                print("❌ 학생 사용자를 찾을 수 없습니다.")
                return False
            
            test_user_id = student[0]
            test_user_name = student[1]
            print(f"✓ 테스트 학생: {test_user_name} (ID: {test_user_id})")
            print()
            
            # 2. 실제 문제 데이터 가져오기
            problems = conn.execute(
                text("""
                    SELECT id, subject, unit, difficulty 
                    FROM problems 
                    WHERE "isActive" = true
                    LIMIT 50
                """)
            ).fetchall()
            
            if not problems:
                print("❌ 활성화된 문제를 찾을 수 없습니다.")
                return False
            
            print(f"✓ 사용 가능한 문제: {len(problems)}개")
            print()
            
            # 3. 기존 테스트 로그 삭제
            print("✓ 기존 테스트 데이터 정리 중...")
            conn.execute(
                text('DELETE FROM attempts WHERE "userId" = :user_id'),
                {"user_id": test_user_id}
            )
            conn.commit()
            
            # 4. 문제 풀이 로그 생성
            print("✓ 문제 풀이 로그 생성 중...")
            
            log_count = 0
            base_time = datetime.now() - timedelta(days=7)
            
            # 각 문제를 랜덤하게 1-3번씩 풀도록 설정
            selected_problems = random.sample(problems, min(30, len(problems)))
            
            for problem in selected_problems:
                problem_id = problem[0]
                difficulty = problem[3]
                
                attempts_count = random.randint(1, 3)
                
                for attempt_num in range(attempts_count):
                    attempt_id = str(uuid.uuid4())
                    
                    # 난이도별 정답률 설정
                    difficulty_rates = {"EASY": 0.85, "MEDIUM": 0.65, "HARD": 0.40}
                    target_rate = difficulty_rates.get(str(difficulty), 0.65)
                    is_correct = random.random() < target_rate
                    
                    # 소요 시간 (초)
                    time_spent = random.randint(30, 300)
                    
                    # 시간 간격을 두고 기록
                    attempt_time = base_time + timedelta(hours=log_count * 2)
                    
                    conn.execute(
                        text("""
                            INSERT INTO attempts (
                                id, "userId", "problemId", "attemptNumber",
                                selected, "isCorrect", "timeSpent", "startedAt",
                                "completedAt", "createdAt", "updatedAt"
                            )
                            VALUES (
                                :id, :user_id, :problem_id, :attempt_number,
                                :selected, :is_correct, :time_spent, :started_at,
                                :completed_at, :created_at, :updated_at
                            )
                        """),
                        {
                            "id": attempt_id,
                            "user_id": test_user_id,
                            "problem_id": problem_id,
                            "attempt_number": attempt_num + 1,
                            "selected": str(random.randint(1, 5)),  # 랜덤 선택
                            "is_correct": is_correct,
                            "time_spent": time_spent,
                            "started_at": attempt_time,
                            "completed_at": attempt_time + timedelta(seconds=time_spent),
                            "created_at": attempt_time,
                            "updated_at": attempt_time
                        }
                    )
                    
                    log_count += 1
            
            conn.commit()
            print(f"✓ 문제 풀이 로그: {log_count}개 생성 완료")
            print()
            
            # 5. 생성된 데이터 확인
            stats = conn.execute(
                text("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN "isCorrect" THEN 1 ELSE 0 END) as correct
                    FROM attempts
                    WHERE "userId" = :user_id
                """),
                {"user_id": test_user_id}
            ).fetchone()
            
            total = stats[0] if stats else 0
            correct = stats[1] if stats else 0
            accuracy = (correct / total * 100) if total > 0 else 0
            
            print("=" * 60)
            print("📊 생성된 데이터 통계")
            print("=" * 60)
            print(f"학생: {test_user_name}")
            print(f"학생 ID: {test_user_id}")
            print(f"총 문제 풀이 수: {total}")
            print(f"정답 수: {correct}")
            print(f"정답률: {accuracy:.2f}%")
            print("=" * 60)
            
            return True
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_test_data()
    sys.exit(0 if success else 1)
