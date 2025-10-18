# 1. 베이스 이미지 설정
# Python 3.11-slim 버전을 사용하여 이미지 크기를 최적화합니다.
FROM python:3.11-slim

# 2. 작업 디렉토리 설정
# 컨테이너 내에서 애플리케이션 코드가 위치할 디렉토리를 설정합니다.
WORKDIR /app

# 3. 시스템 의존성 설치
# psycopg2 라이브러리가 필요로 하는 postgresql-client를 설치합니다.
RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*

# 4. Python 의존성 관리 도구 (uv) 설치
# 프로젝트에서 uv를 사용하므로, pip를 통해 uv를 설치합니다.
RUN pip install uv

# 5. 의존성 파일 복사 및 설치
# 먼저 의존성 관련 파일만 복사하여 Docker의 레이어 캐시를 활용합니다.
# 이렇게 하면, 코드 변경 시마다 의존성을 다시 설치하지 않아 효율적입니다.
COPY pyproject.toml uv.lock* ./
COPY README.md ./

# uv sync를 사용하여 의존성을 설치합니다.
RUN uv sync

# 6. 애플리케이션 코드 복사
# 나머지 애플리케이션 코드를 컨테이너의 작업 디렉토리로 복사합니다.
COPY ./ai-services /app/ai-services
COPY ./backend /app/backend

# 7. 포트 노출
# FastAPI 애플리케이션이 8000번 포트에서 실행되므로, 해당 포트를 노출합니다.
EXPOSE 8000

# 8. 애플리케이션 실행 명령어
# 컨테이너가 시작될 때 실행될 명령어를 정의합니다.
# --host 0.0.0.0 옵션은 컨테이너 외부에서 애플리케이션에 접근할 수 있도록 합니다.
CMD ["/app/.venv/bin/uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
