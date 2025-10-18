# 🎓 Educational AI System

> 교과서 기반 AI 문제 생성 및 학생 분석 시스템
> RAG(Retrieval-Augmented Generation)를 활용한 5지선다 문제 자동 생성 및 학생 데이터 분석

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green.svg)](https://fastapi.tiangolo.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-gpt--5--mini-blue.svg)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-orange.svg)](https://chromadb.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 프로젝트 개요

이 시스템은 **교과서 텍스트를 분석**하여 **맞춤형 5지선다 문제를 자동 생성**하고, 학생의 **문제 풀이 로그를 분석하여 종합 리포트를 생성**하는 AI 시스템입니다. RAG 파이프라인을 통해 정확하고 교육적인 문제를 생성하며, FastAPI를 통해 RESTful API를 제공합니다.

### ✨ 주요 기능

- 📚 **교과서 텍스트 처리**: .txt, .md, .pdf 파일을 지능적으로 청킹
- 🔍 **벡터 임베딩**: OpenAI `text-embedding-ada-002` 기반 고품질 임베딩
- 💾 **벡터 검색**: ChromaDB를 활용한 빠른 유사도 검색
- 🧠 **문제 생성**: `gpt-5-mini`를 사용한 교육적 5지선다 문제 생성
- 👨‍🎓 **학생 리포트 생성**: 학생의 문제 풀이 로그를 분석하여 강점, 약점, 개선 방안을 담은 종합 리포트 생성
- 💡 **힌트 및 태그**: 문제 해결을 위한 학습 보조 힌트와 핵심 태그 자동 생성
- 🚀 **RESTful API**: FastAPI를 활용한 문제 생성 및 학생 분석 API 제공
- 🖥️ **CLI 도구**: 개발 및 디버깅을 위한 명령줄 인터페이스

### 🎯 사용 사례

- **교사**: 교과서 내용 기반 맞춤형 문제 출제 및 학생 학습 상태 분석
- **학생**: 특정 단원에 대한 연습 문제 생성 및 자신의 학습 상태 진단
- **교육기관**: 자동화된 평가 및 피드백 도구 개발
- **에듀테크**: AI 기반 개인 맞춤형 학습 콘텐츠 제작

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

**OCR 기능 사용을 위한 Tesseract 설치 (필수)**

PDF 파일의 텍스트를 추출하기 위해 Tesseract OCR 엔진이 필요합니다. 아래 운영체제에 맞는 안내에 따라 설치해주세요.

- **macOS (Homebrew 사용):**
  ```bash
  brew install tesseract
  brew install tesseract-lang # 한국어 등 추가 언어팩 설치
  ```

- **Ubuntu/Debian:**
  ```bash
  sudo apt update
  sudo apt install tesseract-ocr
  sudo apt install tesseract-ocr-kor # 한국어 언어팩 설치
  ```

- **Windows (Chocolatey 또는 공식 설치 프로그램 사용):**
  - [공식 설치 프로그램 다운로드](https://github.com/UB-Mannheim/tesseract/wiki)
  - 설치 시 "Korean" 언어팩을 반드시 포함하여 설치해야 합니다.

설치 후, `tesseract` 명령어가 시스템 경로에 등록되었는지 확인하세요.

**프로젝트 의존성 설치**

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

`.env.example` 파일을 복사하여 `.env` 파일을 생성하고, OpenAI API 키와 데이터베이스 정보를 설정합니다.

```bash
# 환경 설정 파일 복사
cp .env.example .env
```

`.env` 파일을 열어 아래 내용을 자신의 환경에 맞게 수정합니다.

```dotenv
# OpenAI API 설정
OPENAI_API_KEY=your_openai_api_key_here

# PostgreSQL Database
DB_HOST=localhost
DB_PORT=5432
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_NAME=your_db_name
```

### 3. 백엔드 서버 실행

FastAPI 백엔드 서버를 실행하여 API를 통해 문제를 생성할 수 있습니다.

```bash
# uvicorn을 사용하여 서버 실행
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

서버가 실행되면 브라우저에서 `http://localhost:8000/docs` 로 접속하여 API 문서를 확인하고 테스트할 수 있습니다.

### 4. CLI를 통한 직접 실행

개발 및 디버깅 목적으로 CLI를 직접 사용할 수 있습니다.

**교과서 처리**
```bash
python -m ai-services.src.main process-textbook \
  --file ai-services/data/sample_textbooks/math_unit1.txt \
  --subject 수학 \
  --unit 일차함수
```

**문제 생성**
```bash
python -m ai-services.src.main generate-questions \
  --subject 수학 \
  --unit 일차함수 \
  --difficulty medium \
  --count 1
```

**학생 리포트 생성**
```bash
python -m ai-services.src.main analyze-student --user-id 'user_1234'
```

## 📚 API 엔드포인트

### 문제 생성

- **POST** `/generate-question`
- **설명**: 주어진 조건에 따라 새로운 문제를 생성합니다.
- **요청 본문**:
  ```json
  {
    "subject": "수학",
    "unit": "일차함수",
    "difficulty": "medium",
    "count": 1
  }
  ```
- **성공 응답 (200 OK)**: 생성된 문제 목록 (JSON 배열)

### 학생 성과 분석

- **POST** `/analyze-student-performance`
- **설명**: 특정 학생의 문제 풀이 로그를 분석하여 종합 리포트를 생성합니다.
- **요청 본문**:
  ```json
  {
    "user_id": "user_1234"
  }
  ```
- **성공 응답 (200 OK)**:
  ```json
  {
    "report": "학생 user_1234에 대한 종합 분석 리포트..."
  }
  ```

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