# 1. 베이스 이미지 설정
# Python 3.11-slim 버전을 사용하여 이미지 크기를 최적화합니다.
FROM python:3.11-slim

# 2. 작업 디렉토리 설정
# 컨테이너 내에서 애플리케이션 코드가 위치할 디렉토리를 설정합니다.
WORKDIR /app

# 3. 시스템 의존성 설치
# PDF 처리 및 이미지 처리를 위한 시스템 라이브러리 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    tesseract-ocr \
    tesseract-ocr-kor \
    poppler-utils \
    libpoppler-cpp-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. Python 의존성 관리 도구 (uv) 설치
# 프로젝트에서 uv를 사용하므로, pip를 통해 uv를 설치합니다.
RUN pip install --no-cache-dir uv

# 5. 의존성 파일 복사 및 설치
# 먼저 의존성 관련 파일만 복사하여 Docker의 레이어 캐시를 활용합니다.
# 이렇게 하면, 코드 변경 시마다 의존성을 다시 설치하지 않아 효율적입니다.
COPY pyproject.toml uv.lock* ./
COPY README.md ./

# uv sync를 사용하여 의존성을 설치합니다.
RUN uv sync --no-dev

# 6. 애플리케이션 코드 복사
# 나머지 애플리케이션 코드를 컨테이너의 작업 디렉토리로 복사합니다.
COPY ./ai-services /app/ai-services
COPY ./backend /app/backend
COPY ./airflow /app/airflow
COPY .env.example /app/.env

# 7. 데이터 디렉토리 생성
RUN mkdir -p /app/ai-services/data/vector_db \
    /app/ai-services/data/cache \
    /app/ai-services/data/sample_textbooks \
    /app/db \
    /app/logs

# 8. 포트 노출
# FastAPI: 8000, Airflow Webserver: 8080, Airflow Flower (optional): 5555
EXPOSE 8000 8080

# 9. 환경 변수 설정
ENV PYTHONPATH=/app:${PYTHONPATH}
ENV AIRFLOW_HOME=/app/airflow

# 10. Airflow 초기화 스크립트 생성
RUN echo '#!/bin/bash\n\
if [ ! -f "$AIRFLOW_HOME/airflow.db" ]; then\n\
  echo "Initializing Airflow database..."\n\
  /app/.venv/bin/airflow db migrate\n\
  /app/.venv/bin/airflow users create \\\n\
    --username admin \\\n\
    --password admin \\\n\
    --firstname Admin \\\n\
    --lastname User \\\n\
    --role Admin \\\n\
    --email admin@example.com\n\
fi\n\
' > /app/init_airflow.sh && chmod +x /app/init_airflow.sh

# 11. 헬스체크 스크립트 생성
RUN echo '#!/bin/bash\n\
curl -f http://localhost:8000/health || exit 1\n\
' > /app/healthcheck.sh && chmod +x /app/healthcheck.sh

# 12. 애플리케이션 실행 명령어
# 기본적으로 FastAPI 서버만 실행 (Airflow는 docker-compose에서 별도 컨테이너로 실행)
CMD ["/app/.venv/bin/uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
