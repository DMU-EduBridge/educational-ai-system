# RAG 기반 5지선다 문제 10개 생성 시스템

텍스트 문서를 RAG(Retrieval-Augmented Generation) 처리하여 해당 내용에 기반한 5지선다 문제 10개를 JSON 형태로 생성하는 시스템입니다.

## 🎯 핵심 기능

- **RAG 처리**: 입력 텍스트 문서를 벡터화하고 검색 가능하도록 처리
- **문제 생성**: 문서 내용에 기반한 5지선다 문제 10개 생성
- **JSON 출력**: 구조화된 JSON 형태로 결과 반환
- **다양한 문제 유형**: 개념, 응용, 추론 문제 자동 분배
- **난이도 조절**: 쉬움/보통/어려움 난이도 분산 가능

## 🚀 설치 및 설정

### 1. 환경 설정

```bash
# 1. 필요한 라이브러리 설치
cd ai-services/rag-question-generator
pip install -r requirements.txt

# 2. 환경 변수 설정
cp .env.example .env
```

### 2. OpenAI API 키 설정

`.env` 파일을 열어 OpenAI API 키를 설정하세요:

```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 3. 환경 설정 확인

```bash
python src/main.py --check-env
```

## 📝 사용법

### 기본 사용법

```bash
# 샘플 문서로 문제 생성
python src/main.py -i examples/sample_document.txt -o data/output/questions.json

# 난이도와 문제 수 지정
python src/main.py -i examples/sample_document.txt -n 15 -d hard

# 상세 출력으로 실행
python src/main.py -i examples/sample_document.txt -v
```

### 명령어 옵션

- `-i, --input-file`: 입력 텍스트 문서 경로 (필수)
- `-o, --output-file`: 출력 JSON 파일 경로 (선택사항)
- `-n, --num-questions`: 생성할 문제 수 (1-20, 기본값: 10)
- `-d, --difficulty-mix`: 난이도 구성 (easy/medium/hard/balanced, 기본값: balanced)
- `-v, --verbose`: 상세 출력
- `--check-env`: 환경 설정 상태 확인
- `--setup-env`: 환경 설정 초기화

## 📊 출력 형태

생성된 JSON 파일 구조:

```json
{
  "metadata": {
    "source_document": "sample_document.txt",
    "generated_at": "2024-01-15T10:30:00Z",
    "total_questions": 10,
    "difficulty_distribution": {
      "easy": 3,
      "medium": 4,
      "hard": 3
    }
  },
  "questions": [
    {
      "id": 1,
      "question": "문제 텍스트",
      "options": ["선택지1", "선택지2", "선택지3", "선택지4", "선택지5"],
      "correct_answer": 2,
      "explanation": "정답 해설",
      "difficulty": "medium",
      "type": "concept"
    }
  ]
}
```

## 🏗️ 프로젝트 구조

```
rag-question-generator/
├── src/
│   ├── __init__.py
│   ├── main.py              # 메인 CLI 애플리케이션
│   ├── rag_processor.py     # RAG 처리 모듈
│   ├── question_generator.py # 문제 생성 모듈
│   ├── document_loader.py   # 문서 로더
│   └── config.py           # 설정 관리
├── data/
│   ├── input/              # 입력 문서
│   ├── output/             # 생성된 문제
│   └── vector_db/          # 벡터 데이터베이스
├── examples/
│   ├── sample_document.txt # 샘플 문서
│   └── sample_output.json  # 샘플 출력
├── requirements.txt
├── .env.example
└── README.md
```

## 🔧 설정

### 주요 설정 항목

- `CHUNK_SIZE`: 텍스트 청킹 크기 (기본값: 1000)
- `CHUNK_OVERLAP`: 청크 간 겹치는 부분 (기본값: 200)
- `OPENAI_MODEL`: 사용할 LLM 모델 (기본값: gpt-3.5-turbo)
- `OPENAI_TEMPERATURE`: 생성 창의성 (기본값: 0.7)

### 난이도 분배

- `balanced`: 쉬움 3개, 보통 4개, 어려움 3개
- `easy`: 쉬움 위주 구성
- `medium`: 보통 위주 구성
- `hard`: 어려움 위주 구성

## 🧪 테스트

샘플 문서로 전체 파이프라인 테스트:

```bash
python src/main.py -i examples/sample_document.txt -o test_output.json -v
```

## ⚡ 성능 최적화

- **청킹 최적화**: 문서 내용에 따라 `CHUNK_SIZE` 조정
- **비용 절약**: 필요에 따라 `gpt-3.5-turbo` 대신 더 저렴한 모델 사용
- **배치 처리**: 여러 문서를 한 번에 처리할 때는 벡터 DB 재사용

## 🔍 문제 해결

### 자주 발생하는 오류

1. **API 키 오류**
   ```bash
   python src/main.py --check-env
   ```

2. **메모리 부족**
   - `CHUNK_SIZE` 값을 줄이세요
   - 문서 크기를 확인하세요 (최대 1MB)

3. **문제 생성 실패**
   - 문서 내용이 충분한지 확인 (최소 100자)
   - OpenAI API 할당량을 확인하세요

## 📋 요구사항

- Python 3.8+
- OpenAI API 키
- 최소 2GB RAM (큰 문서 처리 시)
- 인터넷 연결 (OpenAI API 사용)

## 🎉 완료 기준

시스템이 다음 기능들을 모두 지원합니다:

- ✅ 텍스트 문서를 RAG 처리하여 벡터 DB에 저장
- ✅ 문서 내용 기반 5지선다 문제 10개 생성
- ✅ 구조화된 JSON 형태로 결과 출력
- ✅ 다양한 문제 유형 (개념, 응용, 추론) 포함
- ✅ 난이도 분산 (쉬움, 보통, 어려움)
- ✅ CLI 명령어로 실행 가능
- ✅ 환경 설정 및 오류 처리