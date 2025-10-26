# 🎓 Educational AI System

> 교과서 기반 AI 문제 생성 및 주간 학생 리포트 시스템
> RAG(Retrieval-Augmented Generation)와 REST API를 활용한 자동 문제 생성 및 실시간 학생 데이터 분석/리포팅

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green.svg)](https://fastapi.tiangolo.com/)
[![Airflow](https://img.shields.io/badge/Airflow-Workflow-blue.svg)](https://airflow.apache.org/)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini%201.5-blue.svg)](https://ai.google.dev)
[![Langchain](https://img.shields.io/badge/Langchain-Integration-green.svg)](https://python.langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-orange.svg)](https://chromadb.com)
[![SQLite](https://img.shields.io/badge/SQLite-Database-blue.svg)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 프로젝트 개요

이 시스템은 **교과서 텍스트를 분석**하여 **맞춤형 5지선다 문제를 자동 생성**하고, **REST API를 통해 학생의 학습 로그를 실시간으로 분석하여 종합 리포트를 생성**하는 AI 시스템입니다.

### ✨ 주요 기능

- 📚 **교과서 텍스트 처리**: .txt, .md, .pdf 파일을 지능적으로 청킹
- 🧠 **AI 문제 생성**: `Google Gemini 2.5 Flash`를 사용한 교육적 5지선다 문제 생성 (Langchain 통합)
- 🔍 **벡터 검색**: Google Embeddings / 로컬 임베딩을 활용한 의미 기반 문서 검색
- 👨‍🎓 **학습 리포트 API**: REST API를 통해 실시간으로 학생의 학습 로그를 분석하고, 강점, 약점, 개선 방안을 담은 종합 리포트를 생성
- 🚀 **RESTful API**: FastAPI를 활용한 문제 생성 및 리포트 생성 API 제공
- 💬 **AI 챗봇**: WebSocket 기반 실시간 학습 지원 챗봇
- 🖥️ **CLI 도구**: 개발 및 디버깅을 위한 명령줄 인터페이스
- 💰 **비용 효율**: OpenAI 대비 99% 이상 비용 절감

### 🆕 최근 업데이트

- ✅ **리포트 생성 API 추가** (2025.10.25)
  - Airflow → REST API 방식으로 변경
  - 실시간 리포트 생성 지원
  - 프론트엔드 통합 용이
  - 상세 내용: [docs/REPORT_API_GUIDE.md](./docs/REPORT_API_GUIDE.md)

- ✅ **Google Gemini API로 마이그레이션** (2025.10.21)
  - OpenAI → Google Gemini 2.5 Flash
  - Langchain을 통한 통합 구현
  - 99% 이상 비용 절감
  - 상세 내용: [docs/MIGRATION_SUMMARY.md](./docs/MIGRATION_SUMMARY.md)

## 🏗️ 시스템 아키텍처

```
educational-ai-system/
├── airflow/                      # Apache Airflow 설정 및 DAG
│   ├── dags/
│   │   └── weekly_report_dag.py # 주간 리포트 생성 DAG
│   ├── logs/                     # Airflow 로그
│   └── airflow.cfg               # Airflow 설정 파일
├── backend/                      # FastAPI 백엔드
│   └── main.py                   # API 엔드포인트 정의
├── ai-services/                  # AI 서비스 코어
│   ├── src/
│   │   ├── analysis/            # 학생 분석 모듈
│   │   ├── chatbot/             # AI 튜터 챗봇
│   │   ├── evaluation/          # 문제 품질 평가
│   │   ├── models/              # LLM 클라이언트 (Gemini)
│   │   ├── rag/                 # RAG 파이프라인
│   │   └── utils/               # 유틸리티 함수
│   ├── data/                    # 데이터 저장소
│   │   ├── vector_db/           # ChromaDB 벡터 DB
│   │   ├── cache/               # 임베딩 캐시
│   │   └── sample_textbooks/    # 샘플 교과서
│   └── tests/                   # AI 서비스 단위 테스트
├── tests/                       # 통합 테스트 스크립트
│   ├── test_embedding.py        # 임베딩 테스트
│   ├── test_vector_search.py    # 벡터 검색 테스트
│   ├── test_rag_local.py        # RAG 파이프라인 테스트
│   ├── test_question_gen.py     # 문제 생성 테스트
│   ├── test_backend_api.py      # 백엔드 API 테스트
│   └── test_all_units.py        # 전체 통합 테스트
├── scripts/                     # 유틸리티 스크립트
│   └── download_models.py       # 모델 다운로드
├── docs/                        # 프로젝트 문서
│   ├── COMPREHENSIVE_TEST_REPORT.md  # 테스트 보고서
│   ├── ISSUE_RESOLUTION_REPORT.md    # 이슈 해결 보고서
│   ├── GEMINI_MIGRATION.md          # Gemini 마이그레이션 가이드
│   └── DOCKER_GUIDE.md              # Docker 사용 가이드
├── db/                          # SQLite 데이터베이스 (로컬)
├── logs/                        # 애플리케이션 로그
├── Dockerfile                   # Docker 이미지 정의
├── docker-compose.yml           # 다중 컨테이너 구성
├── pyproject.toml               # Python 의존성 관리
├── main.py                      # 메인 진입점
└── .env                         # 환경 변수 설정
```

### 주요 컴포넌트

1. **RAG Pipeline**: 교과서 텍스트를 벡터화하고 의미 기반 검색
2. **Question Generator**: Gemini API를 사용한 5지선다 문제 생성
3. **Student Analyzer**: 학습 로그 분석 및 약점 파악
4. **Report API**: REST API를 통한 실시간 학습 리포트 생성
5. **AI Chatbot**: WebSocket 기반 실시간 학습 지원 챗봇
6. **FastAPI Backend**: RESTful API 서버

## 🚀 빠른 시작

### 1. 사전 요구사항

- Python 3.11+
- Docker & Docker Compose (선택사항)
- Google API Key ([발급 방법](https://makersuite.google.com/app/apikey))

### 2. 설치

#### 방법 A: 로컬 설치 (권장 - 개발용)

```bash
# 1. 저장소 클론
git clone https://github.com/DMU-EduBridge/educational-ai-system.git
cd educational-ai-system

# 2. 가상 환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. UV 패키지 매니저로 의존성 설치
pip install uv
uv sync

# 4. 환경 변수 설정
cp .env.example .env
# .env 파일을 열어 GOOGLE_API_KEY를 설정하세요
```

#### 방법 B: Docker로 설치 (권장 - 프로덕션용)

```bash
# 1. 저장소 클론
git clone https://github.com/DMU-EduBridge/educational-ai-system.git
cd educational-ai-system

# 2. 환경 변수 설정
cp .env.example .env
# .env 파일을 열어 GOOGLE_API_KEY와 데이터베이스 설정을 수정하세요

# 3. Docker Compose로 모든 서비스 실행
docker-compose up -d

# 4. 로그 확인
docker-compose logs -f
```

### 3. 환경 설정

`.env` 파일에서 다음 필수 항목을 설정하세요:

```bash
# Google Gemini API Key (필수)
GOOGLE_API_KEY=your_actual_api_key_here

# 모델 설정
GEMINI_MODEL=gemini-2.5-flash

# 데이터베이스 URL
DATABASE_URL=postgresql://eduai:eduai2025@localhost:5432/educational_ai
```

### 4. 서비스 실행

#### 로컬 환경

```bash
# 가상 환경 활성화
source .venv/bin/activate

# FastAPI 백엔드 서버 실행 (포트 8000)
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 또는 UV를 사용하여 실행
uv run uvicorn main:app --reload --port 8000

# 백그라운드 실행 (선택사항)
nohup uv run uvicorn main:app --reload --port 8000 > ../logs/backend.log 2>&1 &
```

**참고**: Airflow는 현재 선택적 기능입니다. 리포트 생성은 REST API(`/generate-report`)를 통해 실시간으로 가능합니다.

#### Docker 환경

```bash
# 모든 서비스 시작
docker-compose up -d

# 특정 서비스만 시작
docker-compose up -d backend
docker-compose up -d airflow-webserver airflow-scheduler

# 서비스 상태 확인
docker-compose ps

# 로그 확인
docker-compose logs -f backend
docker-compose logs -f airflow-webserver
```

### 5. 접속 URL

- **FastAPI 백엔드**: http://localhost:8000
- **API 문서 (Swagger)**: http://localhost:8000/docs
- **Airflow UI**: http://localhost:8080 (admin/admin)
- **PostgreSQL**: localhost:5432 (Neon 클라우드 DB 사용)

## 📚 API 엔드포인트

### REST API

서버 실행 후 http://localhost:8000/docs 에서 Swagger UI를 통해 모든 API를 테스트할 수 있습니다.

#### 헬스 체크

- **GET** `/`
- **설명**: 서버 상태 확인
- **응답**: `{"status": "ok", "message": "Welcome to the Educational AI System API!"}`

#### 문제 생성

- **POST** `/generate-question`
- **설명**: 주어진 조건에 따라 새로운 5지선다 문제를 생성합니다.
- **요청 본문**:
  ```json
  {
    "subject": "수학",
    "unit": "일차함수",
    "difficulty": "medium",
    "count": 1
  }
  ```
- **응답 예시** (DB Problem 테이블 형식):
  ```json
  [
    {
      "id": null,
      "title": "일차함수의 기울기 구하기",
      "description": "기울기 개념을 이해하는 문제",
      "content": "y = 2x + 3에서 기울기는?",
      "type": "MULTIPLE_CHOICE",
      "difficulty": "MEDIUM",
      "subject": "MATH",
      "gradeLevel": "MIDDLE_3",
      "unit": "일차함수",
      "options": ["1", "2", "3", "4", "5"],
      "correctAnswer": "2",
      "explanation": "일차함수 y=ax+b에서 a가 기울기입니다...",
      "hints": ["일차함수의 일반형을 기억하세요"],
      "tags": ["수학", "일차함수", "기울기"],
      "points": 1,
      "timeLimit": null,
      "isActive": true,
      "isAIGenerated": true,
      "aiGenerationId": "수학_일차함수_medium_1",
      "qualityScore": null,
      "reviewStatus": "PENDING",
      "status": "DRAFT",
      "modelName": "gemini-2.5-flash",
      "tokensUsed": null,
      "costUsd": null,
      "createdAt": "2025-10-25T12:00:00.000Z",
      "updatedAt": "2025-10-25T12:00:00.000Z"
    }
  ]
  ```

#### 학습 리포트 생성 ⭐ NEW

- **POST** `/generate-report`
- **설명**: 학생의 학습 데이터를 실시간으로 분석하여 종합 리포트를 생성합니다.
- **요청 본문**:
  ```json
  {
    "user_id": "cmgp37il10002eg3ztaonfui8"
  }
  ```
- **응답 예시**:
  ```json
  {
    "user_id": "cmgp37il10002eg3ztaonfui8",
    "report_text": "### 학습 보고서\n\n**1. 전반적인 학습 현황 분석**\n총 40개의 문제를 해결했으며...",
    "weakest_unit": "광합성",
    "performance_summary": {
      "total_problems_solved": 40,
      "overall_correct_rate": "72.50%",
      "average_time_spent_seconds": "182.25",
      "performance_by_subject": {...},
      "performance_by_unit": {...},
      "performance_by_difficulty": {...}
    },
    "generated_at": "2025-10-25T21:07:02.490543"
  }
  ```
- **특징**:
  - 실시간 리포트 생성 (12-22초 소요)
  - 과목별/단원별/난이도별 상세 분석
  - 강점, 약점, 개선 방안 제공
  - 리포트당 비용: ~$0.00008 (약 0.1원)
- **상세 문서**: [리포트 API 가이드](./docs/REPORT_API_GUIDE.md)

#### 챗봇 메시지 전송 (REST)

- **POST** `/chat/message`
- **설명**: 챗봇과 단일 메시지를 주고받습니다. 서버가 대화 기록을 관리합니다.
- **요청 본문**:
  ```json
  {
    "user_id": "user_1234",
    "user_message": "일차함수 개념을 다시 설명해줄래?"
  }
  ```
- **성공 응답 (200 OK)**:
  ```json
  {
    "ai_response": "네, 일차함수에 대해 다시 설명해 드릴게요..."
  }
  ```

### WebSocket API

#### 실시간 학습 챗봇

- **WebSocket** `/ws/chat/{user_id}`
- **설명**: 특정 학생을 위한 실시간 대화형 학습 챗봇 세션을 시작합니다. 
- **특징**:
  - DB에 저장된 최신 주간 리포트 기반 개인화된 대화
  - 실시간 양방향 통신
  - 최근 학습 성과 실시간 반영

#### WebSocket 연결 예시

```python
import asyncio
import websockets
import json

async def chat_with_tutor(user_id: str):
    uri = f"ws://localhost:8000/ws/chat/{user_id}"
    
    async with websockets.connect(uri) as websocket:
        # 첫 인사 메시지 수신
        initial_message = await websocket.recv()
        print(f"AI 튜터: {initial_message}")

        while True:
            # 사용자 입력
            user_input = input("나: ")
            if user_input.lower() in ['exit', 'quit', '종료']:
                print("대화를 종료합니다.")
                break

            # 메시지 전송
            await websocket.send(user_input)

            # 튜터 응답 수신
            tutor_response = await websocket.recv()
            print(f"AI 튜터: {tutor_response}")

if __name__ == "__main__":
    # user_1234를 실제 사용자 ID로 변경
    asyncio.run(chat_with_tutor('user_1234'))
```

## 🔧 CLI 도구 사용법

프로젝트는 개발 및 테스트를 위한 CLI 도구를 제공합니다.

### 교과서 처리

```bash
# 교과서 파일을 벡터 DB에 저장
python ai-services/src/main.py process-textbook \
  --file ./data/math_textbook.txt \
  --subject 수학 \
  --unit 일차함수
```

### 문제 생성

```bash
# 문제 생성
python ai-services/src/main.py generate-questions \
  --subject 수학 \
  --unit 일차함수 \
  --difficulty medium \
  --count 5 \
  --output ./questions.json
```

### 시스템 상태 확인

```bash
# 벡터 DB 상태 및 사용량 확인
python ai-services/src/main.py status
```

### 파이프라인 테스트

```bash
# 전체 파이프라인 테스트
python ai-services/src/main.py test-pipeline
```

## 🧪 테스트

### 단위 테스트

```bash
# AI 서비스 단위 테스트
pytest ai-services/tests/

# 특정 테스트 파일 실행
pytest ai-services/tests/test_question_generator.py

# 커버리지 포함
pytest --cov=ai-services/src ai-services/tests/
```

### 통합 테스트

```bash
# 임베딩 테스트
uv run python tests/test_embedding.py
uv run python tests/test_local_embedding.py

# 벡터 DB 및 RAG 테스트
uv run python tests/check_vectordb.py
uv run python tests/test_vector_search.py
uv run python tests/test_rag_local.py
uv run python tests/verify_rag.py

# 문제 생성 테스트
uv run python tests/test_question_gen.py
uv run python tests/test_question_from_db.py

# 백엔드 API 테스트
uv run python tests/test_backend_api.py

# 리포트 생성 API 테스트 ⭐ NEW
uv run python tests/test_report_api.py

# 테스트 데이터 생성 (리포트 테스트용)
uv run python tests/create_test_data.py

# DB 스키마 확인
uv run python tests/check_db_schema.py

# 전체 통합 테스트
uv run python tests/test_all_units.py
```

## 📊 모니터링 및 로깅

### 로그 확인

```bash
# Docker 환경
docker-compose logs -f backend
docker-compose logs -f airflow-scheduler

# 로컬 환경
tail -f logs/app.log
tail -f airflow/logs/scheduler/latest/*.log
```

### API 성능 모니터링

FastAPI는 자동으로 성능 메트릭을 제공합니다:

1. **Swagger UI**: http://localhost:8000/docs
   - 각 엔드포인트의 응답 시간 확인
   - 요청/응답 예시 및 스키마

2. **응답 시간**:
   - 문제 생성: ~8-30초 (RAG 파이프라인 포함)
   - 리포트 생성: ~12-22초 (LLM 분석 포함)
   - 챗봇 메시지: ~2-5초

3. **로그 확인**:
   ```bash
   tail -f logs/backend.log
   ```

## 💰 비용 관리

### 현재 비용 (Google Gemini)

- **LLM (Gemini 2.5 Flash)**:
  - Input: $0.075 / 1M tokens
  - Output: $0.30 / 1M tokens
  
- **Embeddings**: 무료 (2025년 기준)

### 예상 월 비용

| 사용량 | OpenAI 비용 | Gemini 비용 | 절감률 |
|--------|-------------|-------------|--------|
| 1M tokens | ~$250 | ~$0.40 | 99.8% |
| 10M tokens | ~$2,500 | ~$4.00 | 99.8% |

### 실제 사용 비용 (2025.10 기준)

| 기능 | 평균 토큰 | 비용/요청 | 비고 |
|------|----------|----------|------|
| 문제 1개 생성 | ~1,380 tokens | $0.00006 (~0.08원) | RAG 컨텍스트 포함 |
| 리포트 1개 생성 | ~2,300 tokens | $0.00008 (~0.1원) | 통계 분석 포함 |
| 챗봇 메시지 1회 | ~500 tokens | $0.00002 (~0.03원) | 대화 히스토리 포함 |

### 비용 추적

```python
# Python에서 비용 추적
from src.models.llm_client import LLMClient
from src.utils.config import get_settings

settings = get_settings()
llm = LLMClient(
    model_name=settings.gemini_model,
    api_key=settings.google_api_key
)

# 사용량 확인
usage = llm.track_usage()
print(f"총 비용: ${usage['total_cost_usd']:.6f}")
```

## 🐳 Docker 명령어 모음

### 기본 명령어

```bash
# 모든 서비스 시작
docker-compose up -d

# 특정 서비스만 시작
docker-compose up -d backend
docker-compose up -d postgres

# 서비스 중지
docker-compose stop

# 서비스 제거 (볼륨 유지)
docker-compose down

# 서비스 및 볼륨 완전 제거
docker-compose down -v

# 이미지 재빌드
docker-compose build --no-cache
docker-compose up -d --build
```

### 로그 및 디버깅

```bash
# 실시간 로그 확인
docker-compose logs -f

# 특정 서비스 로그
docker-compose logs -f backend
docker-compose logs -f airflow-webserver

# 최근 100줄 로그
docker-compose logs --tail=100 backend

# 컨테이너 접속
docker-compose exec backend bash
docker-compose exec postgres psql -U eduai -d educational_ai
```

### 데이터베이스 관리

```bash
# PostgreSQL 접속
docker-compose exec postgres psql -U eduai -d educational_ai

# 데이터베이스 백업
docker-compose exec postgres pg_dump -U eduai educational_ai > backup.sql

# 데이터베이스 복원
docker-compose exec -T postgres psql -U eduai educational_ai < backup.sql

# 데이터베이스 초기화
docker-compose down -v
docker-compose up -d postgres
```

## 🔒 보안 및 환경 변수

### 환경 변수 설정

프로덕션 환경에서는 다음 변수들을 안전하게 관리하세요:

```bash
# .env 파일 (절대 Git에 커밋하지 마세요!)
GOOGLE_API_KEY=your_actual_key_here
POSTGRES_PASSWORD=strong_random_password
AIRFLOW_SECRET_KEY=generate_secure_random_key
DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

### .gitignore 확인

```bash
# 다음 파일들이 .gitignore에 포함되어 있는지 확인
.env
*.db
*.log
__pycache__/
.venv/
ai-services/data/vector_db/
ai-services/data/cache/
```

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📧 문의

- **프로젝트 저장소**: [GitHub](https://github.com/DMU-EduBridge/educational-ai-system)
- **이슈 트래커**: [Issues](https://github.com/DMU-EduBridge/educational-ai-system/issues)
- **문서**: [Wiki](https://github.com/DMU-EduBridge/educational-ai-system/wiki)

## 🙏 감사의 말

- [Google Gemini API](https://ai.google.dev) - 강력한 LLM 제공
- [Langchain](https://python.langchain.com) - LLM 통합 프레임워크
- [FastAPI](https://fastapi.tiangolo.com) - 고성능 API 프레임워크
- [Apache Airflow](https://airflow.apache.org) - 워크플로우 자동화
- [ChromaDB](https://chromadb.com) - 벡터 데이터베이스

## 📚 추가 문서

### 주요 문서
- **[리포트 API 가이드](./docs/REPORT_API_GUIDE.md)** ⭐ - 학습 리포트 생성 API 사용법
- **[리포트 API 구현 보고서](./docs/REPORT_API_IMPLEMENTATION.md)** - Airflow → REST API 전환 상세
- **[문제 생성 테스트 보고서](./docs/QUESTION_GENERATION_TEST_REPORT.md)** - 문제 생성 기능 테스트 결과
- **[DB 스키마 매핑 가이드](./docs/DB_SCHEMA_MAPPING.md)** - PostgreSQL Problem 테이블 매핑

### 마이그레이션 문서
- [Gemini 마이그레이션](./docs/GEMINI_MIGRATION.md) - OpenAI에서 Gemini로 전환
- [마이그레이션 요약](./docs/MIGRATION_SUMMARY.md) - 변경 사항 상세
- [로컬 임베딩 마이그레이션](./docs/LOCAL_EMBEDDING_MIGRATION.md) - 로컬 임베딩 사용법

### 테스트 문서
- [종합 테스트 보고서](./docs/COMPREHENSIVE_TEST_REPORT.md) - 전체 시스템 테스트 결과
- [이슈 해결 보고서](./docs/ISSUE_RESOLUTION_REPORT.md) - 해결된 이슈 상세

### 기타 문서
- [Docker 가이드](./docs/DOCKER_GUIDE.md) - Docker 사용 방법
- [단원 자동 감지](./docs/UNIT_AUTO_DETECTION_GUIDE.md) - 단원 자동 감지 기능

---

**Made with ❤️ by DMU-EduBridge Team**