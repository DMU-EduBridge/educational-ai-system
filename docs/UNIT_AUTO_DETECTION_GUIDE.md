# 단원 자동 감지 시스템

## 문제 상황

### 이전 구조
- 비상 중3 수학 교과서 PDF를 **"통합교과서"**라는 단일 unit으로 처리
- 벡터 DB의 모든 1,478개 chunk가 `unit: "통합교과서"`로 저장
- "이차함수", "삼각비" 등 특정 단원으로 검색 시 **검색 결과 없음**

### 오류 발생 원인
1. **벡터 DB 필터링**: `vector_store.py`의 `similarity_search_by_embedding()` 메서드에서 `where_clause`로 메타데이터 필터링
2. **unit 필터가 일치하지 않음**: `unit="이차함수"`로 검색했지만 DB에는 `unit="통합교과서"`만 존재
3. **빈 결과**: `retriever.py`에서 `candidate_docs`가 빈 리스트 반환
4. **ValueError**: `question_generator.py`에서 `"No context found for 수학 - 이차함수"` 오류 발생

```python
# vector_store.py - 필터링 로직
where_clause = {"unit": "이차함수"}  # 매칭 안됨!
results = self.collection.query(
    query_embeddings=[query_embedding],
    where=where_clause  # 결과 0개
)
```

## 해결 방법

### 단원 자동 감지 시스템 구현

#### 1. 키워드 기반 단원 분류
```python
UNIT_KEYWORDS = {
    "실수와 그 계산": ["제곱근", "실수", "무리수", "유리수", ...],
    "이차방정식": ["이차방정식", "인수분해", "근의 공식", ...],
    "이차함수": ["이차함수", "포물선", "꼭짓점", "축", ...],
    "삼각비": ["삼각비", "사인", "코사인", "탄젠트", ...],
    "원의 성질": ["원의 성질", "현", "접선", "중심각", ...],
    "통계": ["산포도", "분산", "표준편차", "상관관계", ...]
}
```

#### 2. 자동 감지 알고리즘
```python
def detect_unit(text: str) -> Optional[str]:
    """텍스트 내용 분석으로 단원 감지"""
    scores = {}
    
    for unit, keywords in UNIT_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            count = len(re.findall(keyword, text, re.IGNORECASE))
            score += count
        scores[unit] = score
    
    # 최고 점수 단원 선택
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return None
```

#### 3. 청크별 단원 할당
```python
for chunk_text in chunk_texts:
    detected_unit = detect_unit(chunk_text)
    
    chunk_metadata = {
        'unit': detected_unit or "통합교과서",
        'auto_detected': detected_unit is not None,
        'subject': '수학',
        'source_file': '비상 중3 수학 교과서.pdf'
    }
```

## 실행 결과

### 벡터 DB 재생성
```bash
python ai-services/scripts/reprocess_textbooks_with_unit_detection.py
```

### 단원별 분포
```json
{
  "total_chunks": 224,
  "unit_distribution": {
    "실수와 그 계산": 33,
    "이차방정식": 45,
    "이차함수": 42,
    "삼각비": 26,
    "원의 성질": 32,
    "통계": 41,
    "통합교과서": 5  # 감지 실패한 청크들
  }
}
```

### 모든 단원 검색 성공
```
✅ 실수와 그 계산: 1 문서 검색 성공
✅ 이차방정식: 1 문서 검색 성공
✅ 이차함수: 1 문서 검색 성공
✅ 삼각비: 1 문서 검색 성공
✅ 원의 성질: 1 문서 검색 성공
✅ 통계: 1 문서 검색 성공
```

### 문제 생성 테스트 결과
```
전체: 6개 단원 중 6개 성공
```

## 코드 위치

### 핵심 파일
- **단원 자동 감지 스크립트**: `ai-services/scripts/reprocess_textbooks_with_unit_detection.py`
- **벡터 스토어 필터링**: `ai-services/src/rag/vector_store.py` (163-168번째 줄)
- **문서 검색**: `ai-services/src/rag/retriever.py` (42-70번째 줄)
- **문제 생성**: `ai-services/src/models/question_generator.py` (58-65번째 줄)

### 테스트 스크립트
- **단일 단원 테스트**: `test_backend_api_이차함수.py`
- **전체 단원 테스트**: `test_all_units.py`
- **벡터 DB 확인**: `check_vectordb.py`

## 향후 개선 방안

### 1. 더 정교한 감지 알고리즘
- **TF-IDF**: 키워드 가중치 부여
- **임베딩 유사도**: 단원별 대표 벡터와 비교
- **페이지 번호**: PDF 페이지 구조 분석

### 2. 단원 경계 처리
- **Sliding Window**: 인접 청크 정보 참조
- **중복 할당**: 경계 청크를 여러 단원에 포함
- **계층 구조**: 대단원 > 중단원 > 소단원

### 3. 검증 및 모니터링
- **수동 검증 샘플**: 무작위 10% 청크 검증
- **대시보드**: 단원별 분포 시각화
- **품질 메트릭**: 자동 감지 정확도 추적

## 주의사항

### 재처리 시
1. **기존 벡터 DB 백업**: 실수로 삭제 방지
2. **배치 처리**: 대용량 PDF는 메모리 관리 필요
3. **단원 검증**: 자동 감지 결과 확인

### 프로덕션 배포
1. **백엔드 재시작**: 벡터 DB 변경 시 필수
2. **캐시 클리어**: ChromaDB 인덱스 재구성
3. **모니터링**: 단원별 검색 성공률 추적

## 참고

### 키워드 업데이트
새로운 교과서 추가 시 `UNIT_KEYWORDS` 딕셔너리에 해당 단원의 키워드 추가

### 로그 확인
```bash
# 단원 감지 결과 확인
python ai-services/scripts/reprocess_textbooks_with_unit_detection.py 2>&1 | grep "Unit distribution"

# 벡터 DB 상태 확인
python check_vectordb.py
```

### API 테스트
```bash
# 특정 단원 테스트
python test_backend_api_이차함수.py

# 모든 단원 테스트
python test_all_units.py
```
