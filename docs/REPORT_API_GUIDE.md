# 리포트 생성 API 가이드

**날짜:** 2025-10-25  
**버전:** 1.0

---

## 📋 개요

학생의 학습 데이터를 분석하여 주간 학습 리포트를 생성하는 REST API 엔드포인트입니다.  
기존 Airflow DAG 방식에서 API 방식으로 변경하여, 필요할 때마다 실시간으로 리포트를 생성할 수 있습니다.

---

## 🔄 변경 사항

### 이전 (Airflow)
```python
# 매주 월요일 자동 실행
schedule="0 0 * * 1"
```
- ❌ 스케줄 기반 자동 생성만 가능
- ❌ 실시간 리포트 확인 불가
- ❌ API 통합 어려움

### 현재 (REST API)
```python
POST /generate-report
{
  "user_id": "cmgp37il10002eg3ztaonfui8"
}
```
- ✅ 필요할 때마다 실시간 생성
- ✅ 프론트엔드에서 직접 호출
- ✅ RESTful API 표준 준수

---

## 🚀 API 사용법

### 엔드포인트

```
POST http://localhost:8001/generate-report
```

### 요청 형식

**Headers:**
```
Content-Type: application/json
```

**Body:**
```json
{
  "user_id": "string"  // 필수: 분석할 학생의 ID
}
```

### 응답 형식

**성공 (200 OK):**
```json
{
  "user_id": "cmgp37il10002eg3ztaonfui8",
  "report_text": "### 학습 보고서\n\n**1. 전반적인 학습 현황 분석**\n...",
  "weakest_unit": "광합성",
  "performance_summary": {
    "total_problems_solved": 40,
    "overall_correct_rate": "72.50%",
    "average_time_spent_seconds": "182.25",
    "performance_by_subject": { ... },
    "performance_by_unit": { ... },
    "performance_by_difficulty": { ... }
  },
  "generated_at": "2025-10-25T21:07:02.490543"
}
```

**에러 응답:**

| 상태 코드 | 설명 | 예시 |
|---------|------|------|
| 404 | 학생을 찾을 수 없거나 학습 데이터 없음 | `{"detail": "해당 사용자에 대한 학습 로그를 찾을 수 없습니다."}` |
| 503 | RAG Pipeline 초기화 실패 | `{"detail": "RAG Pipeline is not available."}` |
| 500 | 내부 서버 오류 | `{"detail": "An internal error occurred: ..."}` |

---

## 💻 사용 예시

### Python (requests)

```python
import requests

url = "http://localhost:8001/generate-report"
data = {"user_id": "cmgp37il10002eg3ztaonfui8"}

response = requests.post(url, json=data, timeout=120)

if response.status_code == 200:
    report = response.json()
    print(f"리포트 생성 완료!")
    print(f"취약 단원: {report['weakest_unit']}")
    print(f"정답률: {report['performance_summary']['overall_correct_rate']}")
    print(f"\n{report['report_text']}")
else:
    print(f"오류: {response.status_code} - {response.json()}")
```

### cURL

```bash
curl -X POST http://localhost:8001/generate-report \
  -H "Content-Type: application/json" \
  -d '{"user_id": "cmgp37il10002eg3ztaonfui8"}'
```

### JavaScript (fetch)

```javascript
const generateReport = async (userId) => {
  const response = await fetch('http://localhost:8001/generate-report', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ user_id: userId })
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const report = await response.json();
  console.log('취약 단원:', report.weakest_unit);
  console.log('리포트:', report.report_text);
  
  return report;
};

// 사용
generateReport('cmgp37il10002eg3ztaonfui8')
  .then(report => console.log('리포트 생성 완료'))
  .catch(error => console.error('오류:', error));
```

---

## 📊 응답 데이터 구조

### `report_text`
- **타입**: string
- **설명**: 한국어로 작성된 전체 학습 리포트 (Markdown 형식)
- **포함 내용**:
  1. 전반적인 학습 현황 분석
  2. 강점 (과목별, 난이도별)
  3. 약점 (과목별, 단원별, 난이도별)
  4. 학습 전략 및 개선 권고

### `weakest_unit`
- **타입**: string
- **설명**: 가장 취약한 단원명
- **예시**: "광합성", "이차방정식", "일차함수"

### `performance_summary`
- **타입**: object
- **설명**: 학습 성과 통계 데이터

#### `performance_summary` 필드

| 필드 | 타입 | 설명 | 예시 |
|------|------|------|------|
| `total_problems_solved` | integer | 총 문제 풀이 수 | 40 |
| `overall_correct_rate` | string | 전체 정답률 | "72.50%" |
| `average_time_spent_seconds` | string | 평균 소요 시간(초) | "182.25" |
| `performance_by_subject` | object | 과목별 정오답 비율 | `{"MATH": {true: 0.79, false: 0.21}}` |
| `performance_by_unit` | object | 단원별 정오답 비율 | `{("MATH", "일차함수"): {true: 0.85}}` |
| `performance_by_difficulty` | object | 난이도별 정오답 비율 | `{"EASY": {true: 0.84, false: 0.16}}` |

---

## 🔍 데이터 분석 알고리즘

### 1. 데이터 수집
```sql
SELECT
    a."isCorrect" AS isCorrect,
    a."timeSpent" AS timeSpent,
    p.subject,
    p.unit,
    p.difficulty
FROM attempts a
JOIN problems p ON a."problemId" = p.id
WHERE a."userId" = :user_id;
```

### 2. 통계 계산
- **전체 정답률** = (정답 수 / 총 문제 수) × 100
- **평균 소요 시간** = 총 소요 시간 / 총 문제 수
- **과목별/단원별/난이도별 정답률** = pandas groupby + value_counts

### 3. LLM 분석
- **모델**: Google Gemini 2.5 Flash
- **입력**: 통계 데이터 + 분석 가이드 프롬프트
- **출력**: JSON 형식 (weakest_unit + report_text)

### 4. 응답 생성
- 통계 데이터 + LLM 리포트 결합
- ISO 8601 형식 타임스탬프 추가

---

## ⚡ 성능 및 비용

### 응답 시간
```
데이터 조회: ~1초
통계 계산: ~0.5초
LLM 생성: ~10-20초
총 소요 시간: ~12-22초
```

### API 비용
```
모델: gemini-2.5-flash
프롬프트: ~800 tokens
응답: ~1,500 tokens
총: ~2,300 tokens

비용: ~$0.00008 / 리포트 (약 0.1원)
```

---

## 🧪 테스트

### 1. 테스트 데이터 생성

```bash
cd /path/to/educational-ai-system
uv run python tests/create_test_data.py
```

**출력:**
```
✓ 테스트 학생: 정현 (ID: cmgp37il10002eg3ztaonfui8)
✓ 사용 가능한 문제: 20개
✓ 문제 풀이 로그: 40개 생성 완료
총 문제 풀이 수: 40
정답 수: 29
정답률: 72.50%
```

### 2. API 테스트

```bash
uv run python tests/test_report_api.py
```

**출력:**
```
✅ 성공! 리포트가 생성되었습니다.

📌 취약 단원: 광합성
📊 성적 요약:
  - 총 문제 수: 40
  - 전체 정답률: 72.50%
  - 평균 소요 시간: 182.25초
```

---

## 🔗 관련 코드

### 백엔드 API (`backend/main.py`)

```python
@app.post("/generate-report", summary="학생 학습 리포트 생성", response_model=ReportResponse)
async def generate_report_endpoint(request: ReportRequest) -> Dict[str, Any]:
    """학생의 학습 데이터를 분석하여 주간 학습 리포트를 생성합니다."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline is not available.")
    
    try:
        logger.info(f"Generating report for user: {request.user_id}")
        
        llm_client = pipeline.llm_client
        analyzer = StudentAnalyzer(llm_client)
        report_data = analyzer.analyze(request.user_id)
        
        if "error" in report_data:
            raise HTTPException(status_code=404, detail=report_data["error"])
        
        from datetime import datetime
        response = {
            "user_id": request.user_id,
            "report_text": report_data.get("report_text", ""),
            "weakest_unit": report_data.get("analysis_data", {}).get("weakest_unit", ""),
            "performance_summary": report_data.get("analysis_data", {}).get("performance_summary", {}),
            "generated_at": datetime.now().isoformat()
        }
        
        logger.info(f"Report generated successfully for user: {request.user_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")
```

### 분석 엔진 (`ai-services/src/analysis/student_analyzer.py`)

```python
class StudentAnalyzer:
    """학생의 문제 풀이 로그를 분석하고 종합 리포트를 생성합니다."""
    
    def analyze(self, user_id: str) -> Dict[str, Any]:
        """학생의 학습 로그를 분석하여 최종 리포트를 JSON 형식으로 생성합니다."""
        logs_df = self._fetch_logs(user_id)
        if logs_df.empty:
            return {"error": "해당 사용자에 대한 학습 로그를 찾을 수 없습니다."}

        summary = self._summarize_logs(logs_df)
        if "error" in summary:
            return summary

        prompt = self._generate_prompt(summary)
        structured_report = self.llm_client.generate_structured_response(prompt, response_format="json")
        
        final_output = {
            "report_text": structured_report.get("report_text", ""),
            "analysis_data": {
                "weakest_unit": structured_report.get("weakest_unit", ""),
                "performance_summary": summary
            }
        }
        
        return final_output
```

---

## 📚 데이터베이스 스키마

### attempts 테이블

| 컬럼 | 타입 | 설명 | 필수 |
|------|------|------|------|
| id | text | 시도 ID (UUID) | ✓ |
| userId | text | 학생 ID | ✓ |
| problemId | text | 문제 ID | ✓ |
| attemptNumber | integer | 시도 번호 | ✓ |
| selected | text | 선택한 답 | ✓ |
| isCorrect | boolean | 정답 여부 | ✓ |
| timeSpent | integer | 소요 시간(초) | ✓ |
| startedAt | timestamp | 시작 시간 | - |
| completedAt | timestamp | 완료 시간 | - |
| createdAt | timestamp | 생성 시간 | ✓ |
| updatedAt | timestamp | 업데이트 시간 | ✓ |

### problems 테이블

| 컬럼 | 타입 | 설명 | 필수 |
|------|------|------|------|
| id | text | 문제 ID | ✓ |
| subject | enum | 과목 (MATH, SCIENCE, ENGLISH, etc.) | ✓ |
| unit | text | 단원명 | - |
| difficulty | enum | 난이도 (EASY, MEDIUM, HARD) | ✓ |
| content | text | 문제 내용 | ✓ |

---

## 🚨 주의사항

### 1. 응답 시간
- 리포트 생성은 LLM 호출을 포함하여 **12-22초** 소요
- 프론트엔드에서 **타임아웃을 최소 30초** 이상 설정 권장
- 로딩 인디케이터 필수

### 2. 데이터 요구사항
- 최소 **10개 이상**의 문제 풀이 로그 필요
- 다양한 과목/단원/난이도 분포 권장
- 최근 7일 데이터 분석 권장

### 3. 에러 처리
```javascript
try {
  const report = await generateReport(userId);
} catch (error) {
  if (error.response?.status === 404) {
    alert('학습 데이터가 부족합니다. 더 많은 문제를 풀어주세요.');
  } else if (error.response?.status === 503) {
    alert('서버가 준비 중입니다. 잠시 후 다시 시도해주세요.');
  } else {
    alert('리포트 생성 중 오류가 발생했습니다.');
  }
}
```

### 4. 캐싱 전략
- 동일 학생의 중복 요청 방지 위해 클라이언트 측 캐싱 권장
- 리포트 생성 후 **최소 1시간** 재생성 제한 권장
- 서버 부하 감소 및 비용 절감

---

## 🔮 향후 개선 사항

### 1. 캐싱 구현
```python
# Redis 기반 캐싱
@cache(ttl=3600)  # 1시간 캐시
def get_report(user_id: str):
    return analyzer.analyze(user_id)
```

### 2. 배치 처리
```python
# 여러 학생 동시 분석
POST /generate-reports
{
  "user_ids": ["user1", "user2", "user3"]
}
```

### 3. 웹훅 지원
```python
# 리포트 생성 완료 시 알림
{
  "webhook_url": "https://example.com/webhook",
  "user_id": "user123"
}
```

### 4. PDF 다운로드
```python
GET /download-report/{user_id}
# PDF 파일 반환
```

---

## 📞 문의

- **기술 문의**: GitHub Issues
- **API 문서**: http://localhost:8001/docs (FastAPI Swagger UI)
- **테스트**: `uv run python tests/test_report_api.py`

---

**작성일:** 2025-10-25  
**작성자:** GitHub Copilot  
**버전:** 1.0
