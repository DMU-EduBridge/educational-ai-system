# Educational AI System - Core Services

> 이 모듈은 Educational AI System의 핵심 RAG 파이프라인을 담당합니다.
> 루트 디렉토리의 `main.py`를 통해 통합 실행하는 것을 권장합니다.

## 🔧 주요 기능

- **RAG 파이프라인**: 문서 처리, 임베딩, 벡터 검색
- **문제 생성**: AI 기반 5지선다 문제 자동 생성 (힌트 기능 포함)
- **벡터 데이터베이스**: ChromaDB 기반 지식 저장소
- **데모 도구**: 문제 생성 결과 확인 및 검증

## 📂 모듈 구조

```
ai-services/
├── src/
│   ├── rag/                    # RAG 파이프라인 핵심 모듈
│   │   ├── document_processor.py  # 문서 처리 및 청킹
│   │   ├── embeddings.py          # OpenAI 임베딩 관리
│   │   ├── vector_store.py        # ChromaDB 벡터 저장소
│   │   └── retriever.py           # 컨텍스트 검색 및 랭킹
│   ├── models/                 # AI 모델 관리
│   │   ├── llm_client.py          # OpenAI LLM 클라이언트
│   │   └── question_generator.py  # 문제 생성기 (힌트 기능 포함)
│   ├── utils/                  # 유틸리티 모듈
│   │   ├── config.py              # 설정 관리
│   │   ├── logger.py              # 로깅 시스템
│   │   └── prompts.py             # 프롬프트 템플릿
│   └── main.py                 # CLI 메인 애플리케이션
├── tests/                      # 테스트 코드 (23개 케이스)
├── demo_question_output.py     # 문제 생성 결과 확인 데모
├── demo_batch_questions.py     # 배치 문제 생성 데모
├── question_output_sample.json # 샘플 출력 데이터
├── data/                       # 데이터 저장소
│   ├── sample_textbooks/       # 샘플 교과서 파일
│   └── vector_db/              # ChromaDB 데이터
└── scripts/                    # 유틸리티 스크립트
```

## 🚀 설치 및 설정

### 1. 환경 요구사항

- Python 3.8 이상
- OpenAI API 키

### 2. 의존성 설치

```bash
cd ai-services
pip install -r requirements.txt
```

### 3. 환경 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집하여 OpenAI API 키 설정
# OPENAI_API_KEY=your_actual_api_key_here
```

### 4. 초기 설정 스크립트 실행

```bash
python scripts/setup_environment.py
```

## 💻 사용법

### CLI 명령어 개요

```bash
python -m src.main --help
```

### 1. 교과서 처리

교과서 파일을 처리하여 벡터 데이터베이스에 저장:

```bash
python -m src.main process-textbook \
  --file data/sample_textbooks/math_unit1.txt \
  --subject 수학 \
  --unit 일차함수
```

### 2. 문제 생성

저장된 교과서 내용을 기반으로 5지선다 문제 생성:

```bash
# 단일 문제 생성
python -m src.main generate-questions \
  --subject 수학 \
  --unit 일차함수 \
  --difficulty medium

# 여러 문제 생성
python -m src.main generate-questions \
  --subject 수학 \
  --unit 일차함수 \
  --difficulty medium \
  --count 5 \
  --output questions.json
```

### 3. 시스템 상태 확인

```bash
python -m src.main status
```

### 4. 파이프라인 테스트

```bash
python -m src.main test-pipeline
```

## 📝 사용 예시

### 완전한 워크플로우

```bash
# 1. 수학 교과서 처리
python -m src.main process-textbook \
  --file data/sample_textbooks/math_unit1.txt \
  --subject 수학 \
  --unit 일차함수

# 2. 과학 교과서 처리
python -m src.main process-textbook \
  --file data/sample_textbooks/science_unit1.txt \
  --subject 과학 \
  --unit "물질의 상태"

# 3. 수학 문제 생성 (쉬운 난이도)
python -m src.main generate-questions \
  --subject 수학 \
  --unit 일차함수 \
  --difficulty easy \
  --count 3

# 4. 과학 문제 생성 (어려운 난이도)
python -m src.main generate-questions \
  --subject 과학 \
  --unit "물질의 상태" \
  --difficulty hard \
  --count 2 \
  --output science_questions.json

# 5. 시스템 상태 및 사용량 확인
python -m src.main status
```

## 🔧 설정 옵션

### 주요 설정 파라미터

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `CHUNK_SIZE` | 1000 | 텍스트 청크 크기 (문자 수) |
| `CHUNK_OVERLAP` | 200 | 청크 간 겹치는 문자 수 |
| `RETRIEVAL_K` | 3 | 검색할 문서 수 |
| `OPENAI_TEMPERATURE` | 0.7 | LLM 창의성 수준 (0.0-2.0) |
| `SIMILARITY_THRESHOLD` | 0.7 | 유사도 검색 임계값 |

### 난이도별 문제 생성 가이드

- **easy**: 기본 개념 이해 확인, 단순 암기, 용어 정의
- **medium**: 개념 적용 및 계산, 예제 문제 응용
- **hard**: 복합적 사고 및 응용, 심화 분석, 문제 해결

## 🧪 테스트

### 전체 테스트 실행

```bash
pytest tests/ -v
```

### 특정 모듈 테스트

```bash
# 문서 처리 테스트
pytest tests/test_document_processor.py -v

# 벡터 저장소 테스트
pytest tests/test_vector_store.py -v

# 문제 생성 테스트
pytest tests/test_question_generator.py -v

# 통합 테스트
pytest tests/test_integration.py -v
```

### 테스트 커버리지

```bash
pytest --cov=src tests/
```

## 📊 성능 및 비용

### 예상 비용 (OpenAI API)

- **임베딩 생성**: ~$0.0001 per 1K tokens
- **문제 생성**: ~$0.002 per 1K tokens (GPT-3.5-turbo)

### 예시 비용 계산

1000자 교과서 파일 1개 처리:
- 임베딩: 약 $0.0002
- 문제 3개 생성: 약 $0.006
- **총 비용**: 약 $0.0062

### 성능 지표

- **문서 처리**: 1000자 당 ~2초
- **임베딩 생성**: 100개 청크 당 ~5초
- **문제 생성**: 1개 문제 당 ~10초
- **검색 속도**: 1000개 문서에서 ~0.1초

## 🔍 문제 해결

### 일반적인 문제

1. **OpenAI API 키 오류**
   ```
   Error: Invalid or missing OpenAI API key
   ```
   - `.env` 파일에서 `OPENAI_API_KEY` 확인
   - API 키가 `sk-`로 시작하는지 확인

2. **ChromaDB 권한 오류**
   ```
   Error: Permission denied
   ```
   - `data/vector_db` 디렉토리 권한 확인
   - 디렉토리를 삭제하고 재생성

3. **메모리 부족**
   - `CHUNK_SIZE`를 줄여서 메모리 사용량 감소
   - 배치 크기 조정

### 로그 확인

디버그 모드로 실행하여 상세 로그 확인:

```bash
python -m src.main --debug --verbose generate-questions --subject 수학 --unit 일차함수
```

## 🔮 향후 개발 계획

- [ ] 다양한 문제 유형 지원 (단답형, 서술형)
- [ ] 웹 인터페이스 개발
- [ ] 실시간 문제 생성 API
- [ ] 학습자 수준별 맞춤 문제
- [ ] 문제 품질 자동 평가
- [ ] 다국어 지원

## 🤝 기여하기

1. Fork 프로젝트
2. Feature 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경사항 커밋 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치 푸시 (`git push origin feature/AmazingFeature`)
5. Pull Request 생성

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 지원

문제가 있거나 질문이 있으시면:

- GitHub Issues에서 문제 보고
- 개발팀에 문의

---

**Educational AI System**을 사용해주셔서 감사합니다! 🎓✨