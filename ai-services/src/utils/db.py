from contextlib import contextmanager
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from .config import get_settings


_engine: Optional[Engine] = None


def _build_database_url() -> str:
    """환경설정으로부터 SQLAlchemy 접속 URL을 생성합니다.

    우선순위: DATABASE_URL(예: postgresql+psycopg://...) -> sqlite 파일 경로
    """
    settings = get_settings()
    if settings.database_url:
        return settings.database_url
    # DATABASE_URL 미설정 시 SQLite 파일로 폴백
    return f"sqlite+pysqlite:///{settings.sqlite_db_path}"


def get_engine() -> Engine:
    """애플리케이션 전역에서 재사용할 SQLAlchemy Engine을 반환합니다."""
    global _engine
    if _engine is None:
        database_url = _build_database_url()
        connect_args = {}
        if database_url.startswith("sqlite"):
            # SQLite는 동일 스레드 제한을 풀어야 FastAPI에서 편하게 사용 가능
            connect_args = {"check_same_thread": False}
        _engine = create_engine(database_url, echo=False, future=True, connect_args=connect_args)
    return _engine


@contextmanager
def get_db_connection():
    """SQLite/Postgres(Neon) 모두에서 동작하는 SQLAlchemy 커넥션 컨텍스트를 제공합니다."""
    engine = get_engine()
    conn = None
    try:
        conn = engine.connect()
        yield conn
    except SQLAlchemyError as e:
        print(f"데이터베이스 연결 오류: {e}")
        raise
    finally:
        if conn is not None:
            conn.close()


def init_schema():
    """필요 시 최소 스키마(problems, attempts)를 생성합니다."""
    engine = get_engine()
    ddl_statements = [
        # problems table
        """
        CREATE TABLE IF NOT EXISTS problems (
            id SERIAL PRIMARY KEY,
            subject TEXT NOT NULL,
            unit TEXT NOT NULL,
            difficulty TEXT NOT NULL
        );
        """,
        # attempts table
        """
        CREATE TABLE IF NOT EXISTS attempts (
            id SERIAL PRIMARY KEY,
            userId TEXT NOT NULL,
            problemId INTEGER NOT NULL REFERENCES problems(id),
            isCorrect INTEGER NOT NULL,
            timeSpent REAL NOT NULL
        );
        """,
    ]

    # SQLite 호환: SERIAL을 사용할 수 없으므로 AUTOINCREMENT 사용
    if _build_database_url().startswith("sqlite"):
        ddl_statements = [
            """
            CREATE TABLE IF NOT EXISTS problems (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                unit TEXT NOT NULL,
                difficulty TEXT NOT NULL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                userId TEXT NOT NULL,
                problemId INTEGER NOT NULL,
                isCorrect INTEGER NOT NULL,
                timeSpent REAL NOT NULL,
                FOREIGN KEY(problemId) REFERENCES problems(id)
            );
            """,
        ]

    with engine.begin() as conn:
        for ddl in ddl_statements:
            conn.execute(text(ddl))


def seed_sample_data():
    """분석 기능을 바로 테스트할 수 있도록 최소 샘플 데이터를 삽입합니다."""
    engine = get_engine()
    with engine.begin() as conn:
        # Insert a sample problem if none exists
        result = conn.execute(text("SELECT COUNT(*) FROM problems"))
        count = result.scalar_one()
        if count == 0:
            conn.execute(
                text(
                    "INSERT INTO problems (subject, unit, difficulty) VALUES (:s, :u, :d)"
                ),
                {"s": "수학", "u": "일차함수", "d": "medium"},
            )

        # Ensure there is at least one attempt for demo user
        result = conn.execute(text("SELECT id FROM problems LIMIT 1"))
        problem_id = result.scalar_one()
        conn.execute(
            text(
                """
                INSERT INTO attempts (userId, problemId, isCorrect, timeSpent)
                VALUES (:userId, :problemId, :isCorrect, :timeSpent)
                """
            ),
            {"userId": "demo_user", "problemId": problem_id, "isCorrect": 1, "timeSpent": 12.3},
        )