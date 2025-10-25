# Google Gemini API 마이그레이션 가이드

## 개요
이 프로젝트는 OpenAI API에서 Google Gemini API로 마이그레이션되었습니다. 모든 LLM 기능은 Langchain을 통해 구현되어 있습니다.

## 주요 변경사항

### 1. LLM 모델
- **이전**: OpenAI GPT-3.5/GPT-4
- **현재**: Google Gemini 1.5 Flash / Pro
- **구현**: Langchain의 `ChatGoogleGenerativeAI`

### 2. 임베딩 모델
- **이전**: OpenAI `text-embedding-ada-002`
- **현재**: Google `models/embedding-001`
- **구현**: Langchain의 `GoogleGenerativeAIEmbeddings`

### 3. 환경 변수

#### 새로운 .env 설정
```bash
# Google Gemini API 설정
GOOGLE_API_KEY=your_google_api_key_here

# 모델 설정
GEMINI_MODEL=gemini-1.5-flash
GEMINI_TEMPERATURE=1.0
GEMINI_MAX_TOKENS=20000

# 임베딩 모델 설정
EMBEDDING_MODEL=models/embedding-001
```

### 4. API 키 발급 방법

1. Google AI Studio 방문: https://makersuite.google.com/app/apikey
2. "Get API key" 클릭
3. 새 프로젝트 생성 또는 기존 프로젝트 선택
4. API 키 생성 및 복사
5. `.env` 파일의 `GOOGLE_API_KEY`에 붙여넣기

## 사용 가능한 모델

### LLM 모델
- `gemini-1.5-flash`: 빠르고 비용 효율적 (권장)
- `gemini-1.5-pro`: 높은 성능과 품질
- `gemini-pro`: 이전 버전

### 임베딩 모델
- `models/embedding-001`: 표준 임베딩 모델
- `models/text-embedding-004`: 최신 임베딩 모델

## 비용 비교

### Google Gemini Pricing (2025년 기준)
- **Gemini 1.5 Flash**:
  - Input: $0.075 / 1M tokens
  - Output: $0.30 / 1M tokens
  
- **Gemini 1.5 Pro**:
  - Input: $1.25 / 1M tokens
  - Output: $5.00 / 1M tokens

- **Embedding**:
  - 현재 무료 (변경 가능)

### 비용 절감
OpenAI GPT-4 대비 약 **10배 이상** 저렴합니다.

## 코드 변경사항

### 1. LLMClient (`src/models/llm_client.py`)
```python
# 이전 (OpenAI)
from openai import OpenAI
client = OpenAI(api_key=api_key)

# 현재 (Gemini with Langchain)
from langchain_google_genai import ChatGoogleGenerativeAI
client = ChatGoogleGenerativeAI(
    model=model_name,
    google_api_key=api_key,
    temperature=temperature
)
```

### 2. EmbeddingsManager (`src/rag/embeddings.py`)
```python
# 이전 (OpenAI)
from openai import OpenAI
client = OpenAI(api_key=api_key)
response = client.embeddings.create(model=model, input=texts)

# 현재 (Gemini with Langchain)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
client = GoogleGenerativeAIEmbeddings(
    model=model_name,
    google_api_key=api_key
)
embeddings = client.embed_documents(texts)
```

### 3. Settings (`src/utils/config.py`)
```python
# 새로운 설정 필드
class Settings(BaseSettings):
    google_api_key: str
    gemini_model: str = "gemini-1.5-flash"
    embedding_model: str = "models/embedding-001"
    gemini_temperature: float = 1.0
    gemini_max_tokens: int = 20000
```

## 설치 및 실행

### 1. 패키지 설치
```bash
uv add langchain-google-genai google-generativeai
```

또는

```bash
pip install langchain-google-genai google-generativeai
```

### 2. 환경 변수 설정
`.env` 파일을 생성하고 Google API 키를 설정:
```bash
GOOGLE_API_KEY=your_actual_google_api_key_here
GEMINI_MODEL=gemini-1.5-flash
```

### 3. 애플리케이션 실행
```bash
# 파이프라인 테스트
python ai-services/src/main.py test-pipeline

# 문제 생성
python ai-services/src/main.py generate-questions \
  --subject 수학 \
  --unit "일차함수" \
  --difficulty medium \
  --count 5
```

## 마이그레이션 체크리스트

- [x] Langchain Google Genai 패키지 설치
- [x] LLMClient를 Gemini로 변경
- [x] EmbeddingsManager를 Google Embeddings로 변경
- [x] Settings에 Google API 설정 추가
- [x] .env 파일 업데이트
- [x] main.py에서 초기화 코드 변경
- [ ] Google API 키 발급 및 설정
- [ ] 테스트 실행 및 검증

## 주의사항

1. **API 키 보안**: `.env` 파일을 절대 Git에 커밋하지 마세요
2. **Rate Limits**: Google API의 속도 제한을 확인하세요
3. **토큰 제한**: Gemini 1.5는 더 큰 컨텍스트 창을 지원합니다 (최대 1M 토큰)
4. **호환성**: 일부 OpenAI 특화 기능은 조정이 필요할 수 있습니다

## 문제 해결

### API 키 오류
```
Error: Invalid or missing Google API key
```
**해결**: `.env` 파일에 올바른 `GOOGLE_API_KEY`가 설정되어 있는지 확인

### 임포트 오류
```
ImportError: cannot import name 'ChatGoogleGenerativeAI'
```
**해결**: 패키지 재설치
```bash
uv add langchain-google-genai google-generativeai --force
```

### 응답 형식 오류
Gemini는 OpenAI와 응답 형식이 다를 수 있습니다. JSON 출력의 경우 프롬프트에 명확히 지시하세요.

## 추가 리소스

- [Google AI Studio](https://makersuite.google.com/)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Langchain Google Genai](https://python.langchain.com/docs/integrations/llms/google_ai)
- [Gemini Pricing](https://ai.google.dev/pricing)

## 지원

문제가 발생하면 이슈를 등록해주세요.
