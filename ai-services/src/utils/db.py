import sqlite3
from contextlib import contextmanager
from .config import get_settings

@contextmanager
def get_db_connection():
    """
    컨텍스트 관리자를 사용하여 SQLite 데이터베이스 연결을 가져옵니다.
    """
    settings = get_settings()
    conn = None
    try:
        conn = sqlite3.connect(settings.sqlite_db_path)
        yield conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()