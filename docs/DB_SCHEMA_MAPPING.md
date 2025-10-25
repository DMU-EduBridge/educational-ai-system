# 문제 생성 DB 스키마 매핑 가이드

**날짜:** 2025-10-25  
**버전:** 1.0

---

## 📋 개요

이 문서는 AI가 생성한 문제가 PostgreSQL `Problem` 테이블에 저장될 수 있도록 데이터 구조를 매핑한 내용을 설명합니다.

---

## 🗄️ DB 테이블 구조

### Problem 테이블 스키마

```sql
CREATE TABLE public."Problem" (
    id                text PRIMARY KEY,
    title             text NOT NULL,
    description       text,
    content           text NOT NULL,
    type              public."ProblemType" NOT NULL,
    difficulty        public."ProblemDifficulty" NOT NULL,
    subject           public."Subject" NOT NULL,
    gradeLevel        public."GradeLevel",
    unit              text,
    options           jsonb,
    correctAnswer     text NOT NULL,
    explanation       text,
    hints             jsonb,
    tags              jsonb,
    points            int4 DEFAULT 1 NOT NULL,
    timeLimit         int4,
    isActive          bool DEFAULT true NOT NULL,
    isAIGenerated     bool DEFAULT false NOT NULL,
    aiGenerationId    text,
    qualityScore      float8,
    reviewStatus      public."ReviewStatus" DEFAULT 'PENDING' NOT NULL,
    status            public."ProblemStatus" DEFAULT 'DRAFT' NOT NULL,
    reviewedBy        text,
    reviewedAt        timestamp(3),
    generationProm    text,
    contextChunkId    jsonb,
    generationTime    int4,
    modelName         text,
    tokensUsed        int4,
    costUsd           float8,
    textbookId        text,
    createdBy         text,
    createdAt         timestamp(3) DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updatedAt         timestamp(3) NOT NULL,
    deletedAt         timestamp(3)
);
```

---

## 🔄 데이터 매핑

### Enum 타입 매핑

#### 1. ProblemType (문제 유형)
```python
MULTIPLE_CHOICE = "MULTIPLE_CHOICE"  # 5지선다
SHORT_ANSWER = "SHORT_ANSWER"        # 단답형
ESSAY = "ESSAY"                      # 서술형
```

AI 생성: `"MULTIPLE_CHOICE"` (고정값)

#### 2. ProblemDifficulty (난이도)
```python
difficulty_map = {
    'easy': 'EASY',
    'medium': 'MEDIUM',
    'hard': 'HARD'
}
```

**입력:** `"easy"`, `"medium"`, `"hard"`  
**DB 저장:** `"EASY"`, `"MEDIUM"`, `"HARD"`

#### 3. Subject (과목)
```python
subject_map = {
    '수학': 'MATH',
    '과학': 'SCIENCE',
    '영어': 'ENGLISH',
    '국어': 'KOREAN',
    '사회': 'SOCIAL_STUDIES',
    '역사': 'HISTORY'
}
```

**입력:** `"수학"`, `"과학"` 등  
**DB 저장:** `"MATH"`, `"SCIENCE"` 등

#### 4. GradeLevel (학년)
```python
gradeLevel = "MIDDLE_3"  # 중학교 3학년 (기본값)
```

**가능한 값:**
- `"MIDDLE_1"`, `"MIDDLE_2"`, `"MIDDLE_3"`
- `"HIGH_1"`, `"HIGH_2"`, `"HIGH_3"`

#### 5. ReviewStatus (검토 상태)
```python
reviewStatus = "PENDING"  # 검토 대기 (기본값)
```

**가능한 값:**
- `"PENDING"` - 검토 대기
- `"APPROVED"` - 승인됨
- `"REJECTED"` - 거부됨

#### 6. ProblemStatus (문제 상태)
```python
status = "DRAFT"  # 초안 (기본값)
```

**가능한 값:**
- `"DRAFT"` - 초안
- `"PUBLISHED"` - 게시됨
- `"ARCHIVED"` - 보관됨

---

## 📝 필드 매핑 상세

### 필수 필드

| DB 필드 | AI 생성 필드 | 타입 | 설명 |
|---------|--------------|------|------|
| `id` | - | text | DB에서 UUID 생성 |
| `title` | `title` | text | 문제 제목 |
| `content` | `content` | text | 문제 본문 |
| `type` | - | enum | `"MULTIPLE_CHOICE"` 고정 |
| `difficulty` | `difficulty` | enum | 매핑된 난이도 |
| `subject` | `subject` | enum | 매핑된 과목 |
| `correctAnswer` | `correctAnswer` | text | 정답 텍스트 |
| `isActive` | - | bool | `true` (기본값) |
| `isAIGenerated` | - | bool | `true` (AI 생성) |
| `reviewStatus` | - | enum | `"PENDING"` (기본값) |
| `status` | - | enum | `"DRAFT"` (기본값) |
| `createdAt` | `createdAt` | timestamp | 생성 시간 |
| `updatedAt` | `updatedAt` | timestamp | 업데이트 시간 |

### 선택 필드

| DB 필드 | AI 생성 필드 | 타입 | 설명 |
|---------|--------------|------|------|
| `description` | `description` | text | 문제 설명 |
| `gradeLevel` | - | enum | `"MIDDLE_3"` (기본값) |
| `unit` | `unit` | text | 단원명 |
| `options` | `options` | jsonb | 선택지 배열 |
| `explanation` | `explanation` | text | 해설 |
| `hints` | `hints` | jsonb | 힌트 배열 |
| `tags` | `tags` | jsonb | 태그 배열 |
| `points` | - | int | 1 (기본값) |
| `timeLimit` | - | int | null (기본값) |
| `aiGenerationId` | `aiGenerationId` | text | AI 생성 ID |
| `qualityScore` | - | float | null (추후 평가) |
| `modelName` | `modelName` | text | AI 모델명 |
| `tokensUsed` | `tokensUsed` | int | 사용 토큰 수 |
| `costUsd` | `costUsd` | float | 생성 비용 |

---

## 💡 데이터 변환 예시

### AI 생성 데이터 (Python)

```python
{
    'title': '일차함수의 기울기 구하기',
    'description': '기울기 개념을 이해하는 문제',
    'content': 'y = 2x + 3에서 기울기는?',
    'type': 'MULTIPLE_CHOICE',
    'difficulty': 'EASY',
    'subject': 'MATH',
    'gradeLevel': 'MIDDLE_3',
    'unit': '일차함수',
    'options': ['1', '2', '3', '4', '5'],
    'correctAnswer': '2',
    'explanation': '일차함수 y=ax+b에서 a가 기울기입니다...',
    'hints': ['일차함수의 일반형을 기억하세요'],
    'tags': ['수학', '일차함수', '기울기'],
    'points': 1,
    'timeLimit': None,
    'isActive': True,
    'isAIGenerated': True,
    'aiGenerationId': '수학_일차함수_easy_1',
    'qualityScore': None,
    'reviewStatus': 'PENDING',
    'status': 'DRAFT',
    'modelName': 'gemini-2.5-flash',
    'tokensUsed': None,
    'costUsd': None,
    'createdAt': '2025-10-25T12:00:00.000Z',
    'updatedAt': '2025-10-25T12:00:00.000Z'
}
```

### DB 저장 SQL

```sql
INSERT INTO "Problem" (
    id, title, description, content, type, difficulty, subject, 
    gradeLevel, unit, options, correctAnswer, explanation, 
    hints, tags, points, timeLimit, isActive, isAIGenerated, 
    aiGenerationId, reviewStatus, status, modelName, 
    createdAt, updatedAt
) VALUES (
    uuid_generate_v4(),
    '일차함수의 기울기 구하기',
    '기울기 개념을 이해하는 문제',
    'y = 2x + 3에서 기울기는?',
    'MULTIPLE_CHOICE',
    'EASY',
    'MATH',
    'MIDDLE_3',
    '일차함수',
    '["1", "2", "3", "4", "5"]'::jsonb,
    '2',
    '일차함수 y=ax+b에서 a가 기울기입니다...',
    '["일차함수의 일반형을 기억하세요"]'::jsonb,
    '["수학", "일차함수", "기울기"]'::jsonb,
    1,
    NULL,
    true,
    true,
    '수학_일차함수_easy_1',
    'PENDING',
    'DRAFT',
    'gemini-2.5-flash',
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);
```

---

## 🔍 검증 규칙

### 1. 필수 필드 검증
- `content`: 빈 문자열 불가
- `options`: 정확히 5개의 선택지
- `correctAnswer`: options 중 하나와 일치

### 2. Enum 값 검증
- `type`: `MULTIPLE_CHOICE`, `SHORT_ANSWER`, `ESSAY` 중 하나
- `difficulty`: `EASY`, `MEDIUM`, `HARD` 중 하나
- `subject`: 정의된 Subject enum 값 중 하나
- `reviewStatus`: `PENDING`, `APPROVED`, `REJECTED` 중 하나
- `status`: `DRAFT`, `PUBLISHED`, `ARCHIVED` 중 하나

### 3. 데이터 타입 검증
- `options`, `hints`, `tags`: 유효한 JSON 배열
- `points`: 양의 정수
- `timeLimit`: null 또는 양의 정수
- `qualityScore`: null 또는 0.0~1.0 사이 값

---

## 🚀 백엔드 API 응답 형식

### GET /generate-question 응답

```json
[
  {
    "id": null,
    "title": "일차함수의 기울기 구하기",
    "description": "기울기 개념을 이해하는 문제",
    "content": "y = 2x + 3에서 기울기는?",
    "type": "MULTIPLE_CHOICE",
    "difficulty": "EASY",
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
    "aiGenerationId": "수학_일차함수_easy_1",
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

---

## 📋 체크리스트

### 구현 완료 항목
- [x] Enum 타입 매핑 (difficulty, subject, type 등)
- [x] 필수 필드 매핑
- [x] JSONB 필드 처리 (options, hints, tags)
- [x] correctAnswer를 텍스트로 변환
- [x] 기본값 설정 (isActive, isAIGenerated, status 등)
- [x] 타임스탬프 생성
- [x] 백엔드 API 응답 형식 통일

### 향후 구현 필요
- [ ] tokensUsed 자동 계산
- [ ] costUsd 자동 계산
- [ ] qualityScore 평가 시스템
- [ ] textbookId 연동
- [ ] contextChunkId 추적
- [ ] generationTime 측정

---

## 🔗 관련 문서

- [ISSUE_RESOLUTION_REPORT.md](./ISSUE_RESOLUTION_REPORT.md) - 이슈 해결 보고서
- [COMPREHENSIVE_TEST_REPORT.md](./COMPREHENSIVE_TEST_REPORT.md) - 테스트 보고서

---

**작성자:** GitHub Copilot  
**최종 업데이트:** 2025-10-25
