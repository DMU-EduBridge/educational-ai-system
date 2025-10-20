# 🎓 Educational AI System

> 교과서 기반 AI 문제 생성 및 주간 학생 리포트 시스템
> RAG(Retrieval-Augmented Generation)와 Airflow를 활용한 자동 문제 생성 및 주간 학생 데이터 분석/리포팅

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green.svg)](https://fastapi.tiangolo.com/)
[![Airflow](https://img.shields.io/badge/Airflow-Workflow-blue.svg)](https://airflow.apache.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-gpt--5--mini-blue.svg)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-orange.svg)](https://chromadb.com)
[![SQLite](https://img.shields.io/badge/SQLite-Database-blue.svg)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 프로젝트 개요

이 시스템은 **교과서 텍스트를 분석**하여 **맞춤형 5지선다 문제를 자동 생성**하고, **Apache Airflow를 통해 학생의 주간 학습 로그를 분석하여 종합 리포트를 생성**하고 데이터베이스에 저장하는 AI 시스템입니다.

### ✨ 주요 기능

- 📚 **교과서 텍스트 처리**: .txt, .md, .pdf 파일을 지능적으로 청킹
- 🧠 **AI 문제 생성**: `gpt-5-mini`를 사용한 교육적 5지선다 문제 생성
- 👨‍🎓 **주간 리포트 자동 생성**: Airflow를 사용하여 매주 학생의 학습 로그를 분석하고, 강점, 약점, 개선 방안을 담은 종합 리포트를 생성하여 DB에 저장
- 🚀 **API 제공**: FastAPI를 활용하여 문제 생성 API 제공
- 🖥️ **CLI 도구**: 개발 및 디버깅을 위한 명령줄 인터페이스

## 🏗️ 시스템 아키텍처

```
educational-ai-system/
├── airflow/
│   ├── dags/                 # Airflow DAG 파일
│   ├── logs/                 # Airflow 로그
│   └── plugins/              # Airflow 플러그인
├── backend/                    # FastAPI 백엔드 모듈
├── ai-services/                # 핵심 AI 서비스 모듈
│   ├── src/
│   │   ├── analysis/           # 학생 분석 모듈
│   │   └── ...
│   └── ...
└── ...
```

## 🚀 빠른 시작

### 1. 설치

(이전과 동일)

### 2. 환경 설정

(이전과 동일)

### 3. 백엔드 서버 실행

FastAPI 백엔드 서버는 문제 생성 API만 제공합니다.

```bash
# uvicorn을 사용하여 서버 실행
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 주간 리포트 생성 (Airflow)

주간 리포트 생성은 Airflow를 통해 자동화됩니다. 아래 절차에 따라 Airflow를 실행할 수 있습니다.

**Airflow 초기화 (최초 1회)**

```bash
# .env 파일 생성 (Airflow가 내부적으로 사용)
echo -e "AIRFLOW_UID=$(id -u)" > .env

# Airflow Docker Compose로 DB 초기화
docker-compose -f docker-compose.airflow.yml run --rm airflow-init
```

**Airflow 실행**

```bash
# Airflow 서비스 시작
docker-compose -f docker-compose.airflow.yml up -d
```

Airflow가 실행되면 브라우저에서 `http://localhost:8080` 로 접속하여 Airflow UI를 확인할 수 있습니다. `weekly_student_reports` DAG이 매주 자동으로 실행되어 리포트를 생성하고 `teacher_reports` 테이블에 저장합니다.

## 📚 API 엔드포인트

현재 백엔드는 문제 생성 API만 제공합니다.

- **POST** `/generate-question`
- **설명**: 주어진 조건에 따라 새로운 문제를 생성합니다.

(이후 내용은 이전과 동일)