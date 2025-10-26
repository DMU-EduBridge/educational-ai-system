# Docker 빠른 시작 가이드

이 가이드는 Docker를 사용하여 Educational AI System을 빠르게 실행하는 방법을 설명합니다.

## 📋 사전 요구사항

- Docker 20.10+ 설치
- Docker Compose 2.0+ 설치
- 최소 4GB RAM
- 최소 10GB 디스크 공간

## 🚀 빠른 시작 (5분)

### 1단계: 저장소 클론

```bash
git clone https://github.com/DMU-EduBridge/educational-ai-system.git
cd educational-ai-system
```

### 2단계: 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집
nano .env  # 또는 vim, code 등
```

**필수 설정**:
```bash
## 🔐 환경 변수 설정

`.env` 파일을 생성하고 다음 변수들을 설정하세요:

```bash
# Google API Key (필수)
GOOGLE_API_KEY=your_google_api_key_here

# PostgreSQL 설정
POSTGRES_USER=eduai
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=educational_ai
DATABASE_URL=postgresql://user:password@host:5432/database
```

자세한 설정은 `.env.example` 파일을 참조하세요.
```

### 3단계: 서비스 실행

```bash
# 모든 서비스 시작 (백그라운드)
docker-compose up -d

# 로그 확인
docker-compose logs -f
```

### 4단계: 서비스 확인

- **FastAPI 백엔드**: http://localhost:8000/docs
- **헬스 체크**: http://localhost:8000/health

## 🔧 개별 서비스 관리

### FastAPI 백엔드 실행

```bash
docker-compose up -d postgres backend
```

### 서비스 재시작

```bash
# 특정 서비스 재시작
docker-compose restart backend

# 모든 서비스 재시작
docker-compose restart
```

### 서비스 중지

```bash
# 중지 (데이터 유지)
docker-compose stop

# 제거 (데이터 유지)
docker-compose down

# 완전 제거 (데이터 포함)
docker-compose down -v
```

## 🐛 문제 해결

### 1. 포트 충돌

**증상**: `port is already allocated` 에러

**해결**:
```bash
# 사용 중인 포트 확인
lsof -i :8000
lsof -i :8080
lsof -i :5432

# docker-compose.yml에서 포트 변경
# ports:
#   - "8001:8000"  # 8001로 변경
```

### 2. 메모리 부족

**증상**: 컨테이너가 자주 재시작됨

**해결**:
```bash
# Docker 메모리 할당 증가 (Docker Desktop 설정)
# 또는 서비스 재시작
docker-compose restart backend
```

### 3. API 키 오류

**증상**: `Invalid or missing Google API key`

**해결**:
```bash
# .env 파일 확인
cat .env | grep GOOGLE_API_KEY

# 컨테이너 재시작
docker-compose restart backend
```

### 4. 데이터베이스 연결 실패

**증상**: `connection refused` 또는 `database doesn't exist`

**해결**:
```bash
# PostgreSQL 컨테이너 상태 확인
docker-compose ps postgres

# PostgreSQL 로그 확인
docker-compose logs postgres

# 데이터베이스 재초기화
docker-compose down -v
docker-compose up -d
```

## 📊 로그 및 디버깅

### 실시간 로그 확인

```bash
# 모든 서비스
docker-compose logs -f

# 특정 서비스만
docker-compose logs -f backend
```

### 컨테이너 내부 접속

```bash
# 백엔드 컨테이너
docker-compose exec backend bash

# PostgreSQL
docker-compose exec postgres psql -U eduai -d educational_ai

# Python 인터프리터
docker-compose exec backend /app/.venv/bin/python
```

### 파일 복사

```bash
# 컨테이너 → 호스트
docker-compose cp backend:/app/logs/app.log ./local_log.log

# 호스트 → 컨테이너
docker-compose cp ./data.json backend:/app/data.json
```

## 🔄 업데이트 및 재배포

### 코드 업데이트

```bash
# 최신 코드 가져오기
git pull origin main

# 이미지 재빌드 및 배포
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### 의존성 업데이트

```bash
# pyproject.toml 수정 후
docker-compose build --no-cache backend
docker-compose up -d
```

## 💾 백업 및 복원

### 데이터베이스 백업

```bash
# 백업 생성
docker-compose exec postgres pg_dump -U eduai educational_ai > backup_$(date +%Y%m%d).sql

# 압축 백업
docker-compose exec postgres pg_dump -U eduai educational_ai | gzip > backup_$(date +%Y%m%d).sql.gz
```

### 데이터베이스 복원

```bash
# SQL 파일에서 복원
docker-compose exec -T postgres psql -U eduai educational_ai < backup_20251021.sql

# 압축 파일에서 복원
gunzip -c backup_20251021.sql.gz | docker-compose exec -T postgres psql -U eduai educational_ai
```

### 볼륨 백업

```bash
# 벡터 DB 백업
tar -czf vector_db_backup_$(date +%Y%m%d).tar.gz ai-services/data/vector_db/

# 전체 데이터 백업
docker-compose exec backend tar -czf /tmp/data_backup.tar.gz /app/ai-services/data
docker-compose cp backend:/tmp/data_backup.tar.gz ./data_backup_$(date +%Y%m%d).tar.gz
```

## 🚀 프로덕션 배포

### 환경 변수 보안

```bash
# .env 파일 권한 설정
chmod 600 .env

# Docker Secrets 사용 (권장)
echo "your_password" | docker secret create postgres_password -
```

### 리소스 제한

`docker-compose.yml`에 추가:

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
```

### 헬스체크 설정

모든 서비스에 헬스체크가 구성되어 있습니다:

```bash
# 헬스 상태 확인
docker-compose ps
```

## 📈 모니터링

### 리소스 사용량 확인

```bash
# 실시간 모니터링
docker stats

# 특정 컨테이너
docker stats educational_ai_backend
```

### 디스크 사용량

```bash
# Docker 전체 디스크 사용량
docker system df

# 볼륨 크기 확인
docker volume ls
docker volume inspect educational-ai-system_postgres_data
```

## 🔐 보안 체크리스트

- [ ] `.env` 파일이 `.gitignore`에 포함되어 있음
- [ ] PostgreSQL 비밀번호를 기본값에서 변경
- [ ] 프로덕션에서 DEBUG=false 설정
- [ ] 불필요한 포트 노출 제거
- [ ] 정기적인 백업 스케줄 설정
- [ ] 로그 로테이션 설정

## 📚 추가 리소스

- [Docker 공식 문서](https://docs.docker.com/)
- [Docker Compose 문서](https://docs.docker.com/compose/)
- [프로젝트 README](./README.md)
- [마이그레이션 가이드](./GEMINI_MIGRATION.md)

## 💡 팁

1. **개발 시**: `docker-compose up`으로 포그라운드 실행하여 로그 실시간 확인
2. **프로덕션**: `docker-compose up -d`로 백그라운드 실행
3. **디버깅**: `docker-compose logs -f [service]`로 특정 서비스 로그 추적
4. **클린업**: 주기적으로 `docker system prune -a`로 미사용 리소스 정리

---

**문제가 발생하면 GitHub Issues에 문의하세요!**
