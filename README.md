# 🎓 Educational AI System

> 교과서 기반 AI 문제 생성 시스템  
> RAG(Retrieval-Augmented Generation)를 활용한 5지선다 문제 자동 생성

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green.svg)](https://fastapi.tiangolo.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-blue.svg)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-orange.svg)](https://chromadb.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 프로젝트 개요

이 시스템은 **교과서 텍스트를 분석**하여 **맞춤형 5지선다 문제를 자동 생성**하는 AI 시스템입니다. RAG 파이프라인을 통해 정확하고 교육적인 문제를 생성하며, FastAPI를 통해 RESTful API를 제공합니다.

### ✨ 주요 기능

- 📚 **교과서 텍스트 처리**: .txt, .md 파일을 지능적으로 청킹
- 🔍 **벡터 임베딩**: OpenAI `text-embedding-ada-002` 기반 고품질 임베딩
- 💾 **벡터 검색**: ChromaDB를 활용한 빠른 유사도 검색
- 🧠 **문제 생성**: `GPT-3.5-turbo`를 사용한 교육적 5지선다 문제 생성
- 💡 **힌트 및 태그**: 문제 해결을 위한 학습 보조 힌트와 핵심 태그 자동 생성
- 🚀 **RESTful API**: FastAPI를 활용한 문제 생성 API 제공
- 🖥️ **CLI 도구**: 개발 및 디버깅을 위한 명령줄 인터페이스

### 🎯 사용 사례

- **교사**: 교과서 내용 기반 맞춤형 문제 출제
- **학생**: 특정 단원에 대한 연습 문제 생성
- **교육기관**: 자동화된 평가 도구 개발
- **에듀테크**: AI 기반 학습 콘텐츠 제작

## 🏗️ 시스템 아키텍처

```
educational-ai-system/
├── backend/                    # FastAPI 백엔드 모듈
│   └── main.py                 # API 엔드포인트 및 서버 로직
├── ai-services/                # 핵심 AI 서비스 모듈
│   ├── src/
│   │   ├── rag/                # RAG 파이프라인 핵심 모듈
│   │   ├── models/             # AI 모델 관리 (문제 생성기 포함)
│   │   ├── utils/              # 유틸리티 모듈 (설정, 로거 등)
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

```bash
# 저장소 클론
git clone https://github.com/DMU-EduBridge/educational-ai-system.git
cd educational-ai-system

# 의존성 설치 (uv 또는 pip 사용)
# uv 권장
uv sync

# 또는 pip
pip install -e .
```

### 2. 환경 설정

`.env.example` 파일을 복사하여 `.env` 파일을 생성하고, OpenAI API 키를 설정합니다.

```bash
# 환경 설정 파일 복사
cp .env.example .env

# .env 파일을 열어 API 키 설정
# OPENAI_API_KEY="sk-your-actual-api-key-here"
```

### 3. 백엔드 서버 실행

FastAPI 백엔드 서버를 실행하여 API를 통해 문제를 생성할 수 있습니다.

```bash
# uvicorn을 사용하여 서버 실행
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

서버가 실행되면 브라우저에서 `http://localhost:8000/docs` 로 접속하여 API 문서를 확인하고 테스트할 수 있습니다.

### 4. (옵션) CLI를 통한 직접 실행

개발 및 디버깅 목적으로 CLI를 직접 사용할 수 있습니다.

```bash
# (첫 사용 시) 교과서 처리 및 벡터 DB 생성
python -m ai-services.src.main process-textbook \
  --file ai-services/data/sample_textbooks/math_unit1.txt \
  --subject 수학 \
  --unit 일차함수

# 문제 생성
python -m ai-services.src.main generate-questions \
  --subject 수학 \
  --unit 일차함수 \
  --difficulty medium \
  --count 1
```

## 📝 문제 출력 형식

API 또는 CLI를 통해 생성된 문제는 `problems` 데이터베이스 스키마와 호환되는 다음과 같은 JSON 형식으로 출력됩니다.

```json
{
  "title": "일차함수의 기울기",
  "description": "일차함수 y = ax + b의 형태에서 기울기 a의 의미를 이해하는지 묻는 문제입니다.",
  "content": "일차함수 y = 2x + 3에서 기울기는 무엇인가?",
  "type": "multiple_choice",
  "difficulty": "easy",
  "subject": "수학",
  "gradeLevel": "Middle-1",
  "unit": "일차함수",
  "options": [
    "1",
    "2",
    "3",
    "-2",
    "0"
  ],
  "correctAnswer": "2",
  "explanation": "일차함수 y = ax + b에서 a가 기울기이므로, y = 2x + 3에서 기울기는 2입니다.",
  "hints": [
    "일차함수의 일반형 y = ax + b를 생각해보세요.",
    "x 앞의 계수가 기울기를 의미합니다."
  ],
  "tags": [
    "일차함수",
    "기울기",
    "y절편"
  ],
  "points": 10,
  "timeLimit": 60,
  "isActive": true,
  "isAIGenerated": true,
  "aiGenerationId": "수학_일차함수_easy_1",
  "qualityScore": null,
  "reviewStatus": "pending",
  "reviewedAt": null,
  "generationPrompt": null,
  "contextChunkIds": null,
  "modelName": "gpt-3.5-turbo",
  "createdAt": "2025-09-30T12:00:00.000Z",
  "updatedAt": "2025-09-30T12:00:00.000Z",
  "deletedAt": null
}
```

## 🔄 워크플로우

1.  **📝 교과서 업로드** → 텍스트 파일을 시스템에 입력 (CLI)
2.  **🔪 텍스트 청킹 & 💾 벡터 저장** → 의미 단위로 분할 후 ChromaDB에 저장
3.  **🚀 API 요청** → 클라이언트가 FastAPI 서버에 문제 생성 요청
4.  **🔍 컨텍스트 검색** → 질의와 관련된 내용을 벡터 DB에서 검색
5.  **🧠 문제 생성** → 검색된 컨텍스트를 기반으로 LLM이 문제 생성
6.  **📤 API 응답** → 생성된 문제를 JSON 형식으로 클라이언트에 반환

## 🧪 테스트

```bash
# 전체 테스트 실행
pytest -v

# 특정 테스트 실행
pytest ai-services/tests/test_integration.py -v

# 커버리지 포함 테스트
pytest --cov=ai-services/src ai-services/tests/
```

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: 놀라운 기능 추가'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 👥 개발자

- **DMU-EduBridge** - 김현종
- **연락처**: general.knell@gmail.com
- **GitHub**: [DMU-EduBridge](https://github.com/DMU-EduBridge)

---