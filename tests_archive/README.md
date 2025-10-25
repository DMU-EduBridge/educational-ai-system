# 테스트 아카이브

이 디렉토리는 개발 과정에서 사용되었지만 현재는 사용하지 않는 테스트 파일들을 보관합니다.

## 📁 보관된 파일

### Gemini API 테스트 (중복)
- `test_gemini_direct.py` - Gemini API 직접 호출 테스트
- `test_gemini_integration.py` - Gemini 통합 테스트
- `test_gemini_question.py` - Gemini 문제 생성 테스트

**참고:** 이 기능들은 현재 `tests/test_question_gen.py`에 통합되어 있습니다.

### 디버깅 도구
- `test_json_cleaning.py` - JSON 파싱 디버깅
- `test_llm_only.py` - LLM 단독 테스트
- `debug_paths.py` - 경로 디버깅

**참고:** 이슈 해결 후 더 이상 필요하지 않습니다.

### 특정 단원 테스트
- `test_backend_api_이차함수.py` - 이차함수 단원 특정 테스트
- `test_backend_api_통합교과서.py` - 통합교과서 특정 테스트

**참고:** `tests/test_all_units.py`에서 모든 단원을 포괄적으로 테스트합니다.

## ⚠️ 주의사항

이 파일들은 **참조용**으로만 보관됩니다. 
- 실행을 보장하지 않습니다
- 정기적인 업데이트가 이루어지지 않습니다
- 필요시 검토 후 복원 가능합니다

## 🔄 파일 복원

특정 파일이 필요한 경우:

```bash
# 파일을 루트 또는 tests/ 디렉토리로 이동
mv tests_archive/[filename] tests/

# 또는 복사
cp tests_archive/[filename] tests/
```

---

**정리 날짜:** 2025-10-25  
**담당자:** DMU-EduBridge Team
