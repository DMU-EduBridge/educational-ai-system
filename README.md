# 🎓 Educational AI System

> 교과서 기반 AI 문제 생성 및 학생 분석/학습 시스템
> RAG(Retrieval-Augmented Generation)를 활용한 자동 문제 생성, 학생 데이터 분석, 그리고 대화형 학습 튜터 제공

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green.svg)](https://fastapi.tiangolo.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-gpt--5--mini-blue.svg)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-orange.svg)](https://chromadb.com)
[![SQLite](https://img.shields.io/badge/SQLite-Database-blue.svg)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 프로젝트 개요

이 시스템은 **교과서 텍스트를 분석**하여 **맞춤형 5지선다 문제를 자동 생성**하고, 학생의 **문제 풀이 로그를 분석하여 종합 리포트를 생성**하며, 이를 바탕으로 **대화형 챗봇을 통해 취약점 학습을 지원**하는 통합 AI 시스템입니다. RAG 파이프라인을 통해 정확하고 교육적인 콘텐츠를 생성하며, FastAPI를 통해 RESTful API와 WebSocket API를 제공합니다.

### ✨ 주요 기능

- 📚 **교과서 텍스트 처리**: .txt, .md, .pdf 파일을 지능적으로 청킹
- 🔍 **벡터 임베딩**: OpenAI `text-embedding-ada-002` 기반 고품질 임베딩
- 💾 **벡터 검색**: ChromaDB를 활용한 빠른 유사도 검색
- 🧠 **문제 생성**: `gpt-5-mini`를 사용한 교육적 5지선다 문제 생성
- 👨‍🎓 **학생 리포트 생성**: 학생의 문제 풀이 로그를 분석하여 강점, 약점, 개선 방안을 담은 종합 리포트 생성
- 💬 **대화형 학습 챗봇**: 학생의 취약점을 기반으로 소크라테스식 대화를 통해 학습을 유도하는 AI 튜터
- 🚀 **실시간 API**: FastAPI를 활용한 RESTful API 및 WebSocket API 제공
- 🖥️ **CLI 도구**: 개발 및 디버깅을 위한 명령줄 인터페이스

### 🎯 사용 사례

- **교사**: 교과서 내용 기반 맞춤형 문제 출제 및 학생 학습 상태 자동 분석
- **학생**: 특정 단원에 대한 연습 문제 생성, 자신의 학습 상태 진단 및 AI 튜터와 1:1 보충 학습
- **교육기관**: 자동화된 평가, 피드백, 그리고 개인 맞춤형 학습 지원 도구 개발
- **에듀테크**: AI 기반의 동적이고 상호작용적인 학습 콘텐츠 제작

## 🏗️ 시스템 아키텍처

```
educational-ai-system/
├── backend/                    # FastAPI 백엔드 모듈
│   └── main.py                 # API 엔드포인트 및 서버 로직
├── ai-services/                # 핵심 AI 서비스 모듈
│   ├── src/
│   │   ├── rag/                # RAG 파이프라인 핵심 모듈
│   │   ├── models/             # AI 모델 관리
│   │   ├── analysis/           # 학생 분석 모듈
│   │   ├── chatbot/            # 챗봇 튜터 모듈
│   │   ├── utils/              # 유틸리티 모듈 (설정, 로거, DB 등)
│   │   └── main.py             # CLI 메인 애플리케이션
│   ├── tests/                  # 테스트 코드
│   └── data/                   # 샘플 데이터 및 벡터 DB
├── main.py                     # 통합 실행 파일 (CLI)
├── pyproject.toml              # 프로젝트 설정 및 의존성 관리
├── .env.example                # 환경 설정 예시
└── README.md
```

## 🚀 빠른 시작

### 1. 설치

(이전과 동일)

### 2. 환경 설정

`.env.example` 파일을 복사하여 `.env` 파일을 생성하고, OpenAI API 키와 데이터베이스 정보를 설정합니다.

```bash
# 환경 설정 파일 복사
cp .env.example .env
```

`.env` 파일을 열어 아래 내용을 자신의 환경에 맞게 수정합니다.

```dotenv
# OpenAI API 설정
OPENAI_API_KEY=your_openai_api_key_here

# SQLite Database
SQLITE_DB_PATH=./data/student_logs.db
```

### 3. 백엔드 서버 실행

FastAPI 백엔드 서버를 실행하여 API를 통해 문제를 생성하고 챗봇을 이용할 수 있습니다.

```bash
# uvicorn을 사용하여 서버 실행
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

서버가 실행되면 브라우저에서 `http://localhost:8000/docs` 로 접속하여 API 문서를 확인하고 테스트할 수 있습니다.

### 4. CLI를 통한 직접 실행

(이전과 동일)

## 📚 API 엔드포인트

### REST API

#### 문제 생성

- **POST** `/generate-question`
- **설명**: 주어진 조건에 따라 새로운 문제를 생성합니다.

#### 학생 성과 분석

- **POST** `/analyze-student-performance`
- **설명**: 특정 학생의 문제 풀이 로그를 분석하여 구조화된 종합 리포트를 생성합니다.

#### 챗봇 메시지 전송 (REST)

- **POST** `/chat/message`
- **설명**: 챗봇과 단일 메시지를 주고받습니다. 클라이언트가 대화 기록을 관리해야 합니다.
- **요청 본문**:
  ```json
  {
    "user_id": "user_1234",
    "user_message": "개념을 다시 설명해줄래?",
    "history": [
      { "role": "assistant", "content": "안녕하세요! ..." },
      { "role": "user", "content": "문제를 풀어볼게." },
      { "role": "assistant", "content": "네, 좋습니다. 다음 문제입니다..." }
    ]
  }
  ```
- **성공 응답 (200 OK)**:
  ```json
  {
    "ai_response": "네, 개념을 다시 설명해 드릴게요...",
    "updated_history": [ ... ]
  }
  ```

### WebSocket API

#### 학습 챗봇

- **WebSocket** `/ws/chat/{user_id}`
- **설명**: 특정 학생을 위한 대화형 학습 챗봇 세션을 시작합니다.
- **연결**: 클라이언트는 이 엔드포인트로 WebSocket 연결을 시작합니다.
- **메시지 흐름**:
  1.  연결이 수립되면, 서버는 학생의 취약 단원을 기반으로 첫 인사 메시지를 보냅니다.
  2.  클라이언트는 사용자 입력을 텍스트 메시지로 서버에 전송합니다.
  3.  서버는 LLM을 통해 응답을 생성하여 클라이언트에 다시 전송합니다.
  4.  이 과정은 연결이 종료될 때까지 반복됩니다.

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

## 🧪 테스트

(이전과 동일)

## 🤝 기여하기

(이전과 동일)

## 📄 라이선스

(이전과 동일)

## 👥 개발자

(이전과 동일)

---