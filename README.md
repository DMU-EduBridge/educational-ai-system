# 🎓 Educational AI System

> 교과서 기반 AI 문제 생성 및 주간 학생 리포트 시스템
> RAG(Retrieval-Augmented Generation)와 Airflow를 활용한 자동 문제 생성 및 주간 학생 데이터 분석/리포팅

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green.svg)](https://fastapi.tiangolo.com/)
[![Airflow](https://img.shields.io/badge/Airflow-Workflow-blue.svg)](https://airflow.apache.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-gpt--5--mini-blue.svg)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-orange.svg)](https://chromadb.com)
[![SQLite](https://img.shields.io/badge/SQLite-Database-blue.svg)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 프로젝트 개요

이 시스템은 **교과서 텍스트를 분석**하여 **맞춤형 5지선다 문제를 자동 생성**하고, **Apache Airflow를 통해 학생의 주간 학습 로그를 분석하여 종합 리포트를 생성**하고 데이터베이스에 저장하는 AI 시스템입니다.

### ✨ 주요 기능

- 📚 **교과서 텍스트 처리**: .txt, .md, .pdf 파일을 지능적으로 청킹
- 🧠 **AI 문제 생성**: `gpt-5-mini`를 사용한 교육적 5지선다 문제 생성
- 👨‍🎓 **주간 리포트 자동 생성**: Airflow를 사용하여 매주 학생의 학습 로그를 분석하고, 강점, 약점, 개선 방안을 담은 종합 리포트를 생성하여 `teacher_reports` DB 테이블에 저장
- 🚀 **API 제공**: FastAPI를 활용하여 문제 생성 API 제공
- 🖥️ **CLI 도구**: 개발 및 디버깅을 위한 명령줄 인터페이스

## 🏗️ 시스템 아키텍처

```
educational-ai-system/
├── airflow/
│   ├── dags/
│   │   └── weekly_report_dag.py
│   ├── logs/
│   └── airflow.cfg
├── backend/
│   └── main.py
├── ai_services/
│   ├── src/
│   │   ├── analysis/
│   │   ├── models/
│   │   └── ...
│   └── tests/
├── pyproject.toml
└── README.md
```

## 🚀 빠른 시작

### 1. 설치

(이전과 동일)

### 2. 환경 설정

(이전과 동일)

### 3. 백엔드 서버 실행

FastAPI 백엔드 서버는 문제 생성 API만 제공합니다.

```bash
# uvicorn을 사용하여 서버 실행
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 주간 리포트 생성 (Airflow)

주간 리포트 생성은 Airflow를 통해 자동화됩니다. 아래 절차에 따라 로컬 환경에서 Airflow를 설정하고 실행할 수 있습니다.

**Airflow 설치 및 초기화 (최초 1회)**

```bash
# 1. 프로젝트 의존성 설치 (apache-airflow 포함)
pip install -e .

# 2. Airflow 홈 디렉토리 설정 및 DB 초기화
export AIRFLOW_HOME=$(pwd)/airflow
airflow db migrate

# 3. Airflow UI 접속을 위한 사용자 생성
airflow users create --username admin --password admin --firstname Anonymous --lastname User --role Admin --email admin@example.com
```

**Airflow 실행**

```bash
# Airflow 홈 디렉토리 설정
export AIRFLOW_HOME=$(pwd)/airflow

# Airflow 웹서버 실행 (백그라운드)
airflow webserver --port 8080 &

# Airflow 스케줄러 실행 (백그라운드)
airflow scheduler &
```

**Airflow UI 설정**

1.  Airflow가 실행되면 브라우저에서 `http://localhost:8080` 로 접속하여 위에서 생성한 계정(admin/admin)으로 로그인합니다.
2.  **Admin > Connections** 메뉴로 이동하여, 프로젝트의 데이터베이스(PostgreSQL) 연결 정보를 설정합니다. `get_db_connection()` 함수가 사용하는 기본 연결 ID(`postgres_default`) 또는 코드에 명시된 다른 ID로 연결을 생성해야 합니다.
3.  `weekly_learning_report` DAG이 목록에 나타나고, 스케줄에 따라 (매주 월요일 0시) 자동으로 실행됩니다.

## 📚 API 엔드포인트

### REST API

#### 문제 생성

- **POST** `/generate-question`
- **설명**: 주어진 조건에 따라 새로운 문제를 생성합니다.

#### 챗봇 메시지 전송 (REST)

- **POST** `/chat/message`
- **설명**: 챗봇과 단일 메시지를 주고받습니다. 서버가 대화 기록을 관리합니다.
- **요청 본문**:
  ```json
  {
    "user_id": "user_1234",
    "user_message": "개념을 다시 설명해줄래?"
  }
  ```
- **성공 응답 (200 OK)**:
  ```json
  {
    "ai_response": "네, 개념을 다시 설명해 드릴게요..."
  }
  ```

### WebSocket API

#### 학습 챗봇

- **WebSocket** `/ws/chat/{user_id}`
- **설명**: 특정 학생을 위한 실시간 대화형 학습 챗봇 세션을 시작합니다. 챗봇은 DB에 저장된 가장 최신 주간 리포트를 기반으로 대화를 시작합니다.

## 💬 챗봇 사용법 (테스트용)

백엔드 서버가 실행 중인 상태에서, 아래의 Python 코드를 사용하여 챗봇과 상호작용할 수 있습니다. `websockets` 라이브러리가 설치되어 있어야 합니다 (`pip install websockets`).

```python
import asyncio
import websockets

async def chat_with_tutor(user_id: str):
    uri = f"ws://localhost:8000/ws/chat/{user_id}"
    async with websockets.connect(uri) as websocket:
        # 첫 인사 메시지 수신
        initial_message = await websocket.recv()
        print(f"AI 튜터: {initial_message}")

        while True:
            # 사용자 입력
            user_input = input("나: ")
            if user_input.lower() == 'exit':
                print("대화를 종료합니다.")
                break

            await websocket.send(user_input)

            # 튜터 응답 수신
            tutor_response = await websocket.recv()
            print(f"AI 튜터: {tutor_response}")

if __name__ == "__main__":
    # 'user_1234'를 실제 테스트하고 싶은 학생 ID로 변경하세요.
    asyncio.run(chat_with_tutor('user_1234'))
```

(이후 내용은 이전과 동일)