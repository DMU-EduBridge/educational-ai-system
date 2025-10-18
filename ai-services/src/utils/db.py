import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
from typing import Optional

from .config import get_settings, Settings

# 전역 연결 풀
_connection_pool: Optional[psycopg2.pool.SimpleConnectionPool] = None

def init_connection_pool(settings: Optional[Settings] = None):
    """데이터베이스 연결 풀 초기화"""
    global _connection_pool
    if _connection_pool is None:
        if settings is None:
            settings = get_settings()
        
        try:
            _connection_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=settings.db_host,
                port=settings.db_port,
                user=settings.db_user,
                password=settings.db_password,
                dbname=settings.db_name,
            )
            print("Database connection pool initialized.")
        except psycopg2.OperationalError as e:
            print(f"Error initializing connection pool: {e}")
            _connection_pool = None
            raise

@contextmanager
def get_db_connection():
    """컨텍스트 관리자를 사용하여 연결 풀에서 연결을 가져옵니다."""
    if _connection_pool is None:
        raise ConnectionError("Connection pool is not initialized. Call init_connection_pool() first.")
    
    conn = None
    try:
        conn = _connection_pool.getconn()
        yield conn
    finally:
        if conn:
            _connection_pool.putconn(conn)

def close_connection_pool():
    """애플리케케이션 종료 시 연결 풀을 닫습니다."""
    global _connection_pool
    if _connection_pool:
        _connection_pool.closeall()
        _connection_pool = None
        print("Database connection pool closed.")
