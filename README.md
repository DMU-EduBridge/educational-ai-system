# 🎓 Educational AI System

> 중학교 교과서 기반 AI 문제 생성 시스템  
> RAG(Retrieval-Augmented Generation)를 활용한 5지선다 문제 자동 생성

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-green.svg)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-orange.svg)](https://chromadb.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 프로젝트 개요

이 시스템은 **중학교 교과서 텍스트를 분석**하여 **맞춤형 5지선다 문제를 자동 생성**하는 AI 시스템입니다.

### ✨ 주요 기능

- 📚 **교과서 텍스트 처리**: .txt, .md 파일을 지능적으로 청킹
- 🔍 **벡터 임베딩**: OpenAI text-embedding-ada-002 기반 고품질 임베딩
- 💾 **벡터 검색**: ChromaDB를 활용한 빠른 유사도 검색
- 🧠 **문제 생성**: GPT-3.5-turbo를 사용한 교육적 5지선다 문제 생성
- 🖥️ **CLI 도구**: 직관적인 명령줄 인터페이스
- 🧪 **완전한 테스트**: 포괄적인 테스트 커버리지

### 🎯 사용 사례

- **교사**: 교과서 내용 기반 맞춤형 문제 출제
- **학생**: 특정 단원에 대한 연습 문제 생성
- **교육기관**: 자동화된 평가 도구 개발
- **에듀테크**: AI 기반 학습 콘텐츠 제작

## 🏗️ 시스템 아키텍처

```
educational-ai-system/
├── main.py                     # 통합 실행 파일
├── pyproject.toml              # 프로젝트 설정
├── .env.example                # 환경 설정 예시
├── ai-services/                # 핵심 AI 서비스
│   ├── src/
│   │   ├── rag/                # RAG 파이프라인 핵심 모듈
│   │   │   ├── document_processor.py  # 문서 처리 및 청킹
│   │   │   ├── embeddings.py          # OpenAI 임베딩 관리
│   │   │   ├── vector_store.py        # ChromaDB 벡터 저장소
│   │   │   └── retriever.py           # 컨텍스트 검색 및 랭킹
│   │   ├── models/             # AI 모델 관리
│   │   │   ├── llm_client.py          # OpenAI LLM 클라이언트
│   │   │   └── question_generator.py  # 문제 생성기
│   │   ├── utils/              # 유틸리티 모듈
│   │   │   ├── config.py              # 설정 관리
│   │   │   ├── logger.py              # 로깅 시스템
│   │   │   └── prompts.py             # 프롬프트 템플릿
│   │   └── main.py             # CLI 메인 애플리케이션
│   ├── tests/                  # 테스트 코드
│   ├── data/                   # 데이터 저장소
│   │   ├── sample_textbooks/   # 샘플 교과서 파일
│   │   ├── vector_db/          # ChromaDB 데이터
│   │   └── cache/              # 캐시 데이터
│   └── scripts/                # 유틸리티 스크립트
└── rag-question-generator/     # 레거시 코드 (제거 예정)
```

## 🚀 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/DMU-EduBridge/educational-ai-system.git
cd educational-ai-system

# 의존성 설치 (uv 권장)
uv sync

# 또는 pip 사용
pip install -e .
```

### 2. 환경 설정

```bash
# 환경 설정 파일 복사
cp .env.example .env

# .env 파일에서 OpenAI API 키 설정
# OPENAI_API_KEY=sk-your-actual-api-key-here

# 시스템 정보 확인
python main.py info

# 환경 설정 초기화
python main.py setup-env
```

```bash
# ai-services/.env 파일 생성
cd ai-services
cp .env.example .env

# .env 파일에 API 키 추가
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 3. 교과서 처리 및 문제 생성

```bash
cd ai-services

# 교과서 텍스트 처리
python -m src.main process-textbook \
  --file data/sample_textbooks/math_unit1.txt \
  --subject 수학 \
  --unit 일차함수

# 문제 생성
python -m src.main generate-questions \
  --subject 수학 \
  --unit 일차함수 \
  --difficulty medium \
  --count 3
```

## 🏗️ 시스템 아키텍처

```
educational-ai-system/
├── ai-services/                # 🎯 AI 서비스 핵심 모듈
│   ├── src/
│   │   ├── rag/               # RAG 파이프라인
│   │   ├── models/            # AI 모델 관리
│   │   ├── utils/             # 유틸리티
│   │   └── main.py           # CLI 진입점
│   ├── data/                  # 데이터 저장소
│   ├── tests/                 # 테스트 스위트
│   └── scripts/               # 유틸리티 스크립트
├── main.py                    # 📍 프로젝트 진입점
├── pyproject.toml            # 📦 의존성 관리
└── README.md                 # 📖 이 파일
```

## 🔄 워크플로우

1. **📝 교과서 업로드** → 텍스트 파일을 시스템에 입력
2. **🔪 텍스트 청킹** → 의미 단위로 텍스트 분할
3. **🧮 벡터 변환** → OpenAI API로 임베딩 생성
4. **💾 벡터 저장** → ChromaDB에 저장
5. **🔍 컨텍스트 검색** → 질의 관련 내용 검색
6. **🧠 문제 생성** → GPT 모델로 5지선다 문제 생성

## 📊 성능 및 비용

- **처리 속도**: 1000자 당 ~2초
- **문제 생성**: 1개 문제 당 ~10초
- **예상 비용**: 1000자 교과서 + 문제 3개 ≈ $0.006
- **정확도**: 58개 테스트 케이스 100% 통과

## 🧪 테스트

```bash
# 전체 테스트 실행
cd ai-services
pytest tests/ -v

# 특정 테스트 실행
pytest tests/test_integration.py -v

# 커버리지 포함 테스트
pytest --cov=src tests/
```

## 📚 더 자세한 정보

- **[AI Services 상세 문서](ai-services/README.md)** - 상세한 구현 가이드
- **[API 문서](ai-services/docs/)** - 함수 및 클래스 레퍼런스
- **[설정 가이드](ai-services/.env.example)** - 환경 변수 설정
- **[예제 데이터](ai-services/data/sample_textbooks/)** - 샘플 교과서 파일

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: 놀라운 기능 추가'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 👥 개발자

- **DMU-EduBridge** - 김현종
- **연락처**: general.knell@gmail.com.com

---

</div>
