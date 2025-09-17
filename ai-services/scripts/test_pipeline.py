#!/usr/bin/env python3
"""
파이프라인 테스트 스크립트
Educational AI System의 전체 기능을 테스트합니다.
"""

import os
import sys
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List
import time

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_application_logger
from src.utils.config import get_test_settings
from src.main import RAGPipeline


class PipelineTester:
    """파이프라인 테스트 클래스"""

    def __init__(self):
        self.logger = setup_application_logger("pipeline_test")
        self.project_root = Path(__file__).parent.parent
        self.test_results = {}
        self.temp_dir = None

    def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        self.logger.info("파이프라인 전체 테스트를 시작합니다...")

        try:
            # 임시 디렉토리 생성
            self.temp_dir = tempfile.mkdtemp(prefix="pipeline_test_")
            self.logger.info(f"테스트 디렉토리: {self.temp_dir}")

            # 테스트 설정
            settings = self._prepare_test_settings()

            # 테스트 실행
            tests = [
                ("환경 설정", self.test_environment),
                ("파이프라인 초기화", lambda: self.test_pipeline_initialization(settings)),
                ("문서 처리", self.test_document_processing),
                ("벡터 저장", self.test_vector_storage),
                ("컨텍스트 검색", self.test_context_retrieval),
                ("문제 생성", self.test_question_generation),
                ("배치 처리", self.test_batch_processing),
                ("성능 측정", self.test_performance),
                ("에러 처리", self.test_error_handling)
            ]

            for test_name, test_func in tests:
                self.logger.info(f"테스트 실행 중: {test_name}")
                try:
                    start_time = time.time()
                    result = test_func()
                    end_time = time.time()

                    self.test_results[test_name] = {
                        'status': 'success',
                        'result': result,
                        'duration': end_time - start_time
                    }
                    self.logger.info(f"✅ {test_name} 성공 ({end_time - start_time:.2f}초)")

                except Exception as e:
                    self.test_results[test_name] = {
                        'status': 'failed',
                        'error': str(e),
                        'duration': 0
                    }
                    self.logger.error(f"❌ {test_name} 실패: {str(e)}")

            # 결과 요약
            self._print_test_summary()

            return self.test_results

        finally:
            # 임시 디렉토리 정리
            if self.temp_dir and Path(self.temp_dir).exists():
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _prepare_test_settings(self):
        """테스트 설정 준비"""
        settings = get_test_settings()
        settings.chroma_db_path = str(Path(self.temp_dir) / "vector_db")
        settings.cache_dir = str(Path(self.temp_dir) / "cache")

        # OpenAI API 키 확인
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == 'your_openai_api_key_here':
            self.logger.warning("OpenAI API 키가 설정되지 않았습니다. Mock 모드로 실행됩니다.")
            settings.openai_api_key = "sk-test-key-for-testing"
        else:
            settings.openai_api_key = api_key

        return settings

    def test_environment(self) -> Dict[str, Any]:
        """환경 테스트"""
        result = {}

        # Python 버전 확인
        result['python_version'] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        # 필수 패키지 확인
        required_packages = ['openai', 'chromadb', 'click', 'pydantic', 'tiktoken']
        missing_packages = []

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        result['missing_packages'] = missing_packages

        # 디렉토리 접근 권한 확인
        test_file = Path(self.temp_dir) / "access_test.txt"
        try:
            test_file.write_text("test")
            test_file.unlink()
            result['file_access'] = True
        except Exception as e:
            result['file_access'] = False
            result['file_access_error'] = str(e)

        if missing_packages:
            raise Exception(f"필수 패키지 누락: {missing_packages}")

        return result

    def test_pipeline_initialization(self, settings) -> Dict[str, Any]:
        """파이프라인 초기화 테스트"""
        from unittest.mock import patch

        # OpenAI API 호출을 Mock으로 대체
        with patch('src.rag.embeddings.openai.OpenAI'), \
             patch('src.models.llm_client.openai.OpenAI'):

            pipeline = RAGPipeline(settings)

            result = {
                'components_initialized': True,
                'vector_store_ready': pipeline.vector_store is not None,
                'llm_client_ready': pipeline.llm_client is not None,
                'question_generator_ready': pipeline.question_generator is not None
            }

            # 저장 변수
            self.pipeline = pipeline

            return result

    def test_document_processing(self) -> Dict[str, Any]:
        """문서 처리 테스트"""
        # 테스트 문서 생성
        test_content = """일차함수의 정의

일차함수는 y = ax + b (a ≠ 0) 형태로 나타낼 수 있는 함수입니다.
여기서 a는 기울기, b는 y절편을 나타냅니다.

일차함수의 그래프는 직선입니다.
기울기가 양수이면 우상향하고, 음수이면 우하향합니다.

예시:
y = 2x + 3 (기울기 2, y절편 3)
y = -x + 5 (기울기 -1, y절편 5)"""

        test_file = Path(self.temp_dir) / "test_textbook.txt"
        test_file.write_text(test_content, encoding='utf-8')

        # 문서 처리
        processor = self.pipeline.document_processor
        documents = processor.load_textbook(str(test_file), "수학", "일차함수")

        result = {
            'total_documents': len(documents),
            'has_metadata': all('subject' in doc.metadata for doc in documents),
            'content_preserved': any('일차함수' in doc.content for doc in documents),
            'chunking_worked': len(documents) > 0
        }

        # 저장
        self.test_documents = documents

        return result

    def test_vector_storage(self) -> Dict[str, Any]:
        """벡터 저장 테스트"""
        from unittest.mock import patch, MagicMock

        # 임베딩 생성 Mock
        with patch.object(self.pipeline.embeddings_manager, 'generate_embeddings') as mock_embed:
            mock_embed.return_value = [[0.1] * 1536 for _ in self.test_documents]

            # 벡터 저장소에 저장
            embeddings = mock_embed.return_value
            success = self.pipeline.vector_store.add_documents(self.test_documents, embeddings)

            # 저장 확인
            info = self.pipeline.vector_store.get_collection_info()

            result = {
                'storage_success': success,
                'stored_documents': info['total_documents'],
                'subjects_stored': len(info['subjects']),
                'units_stored': len(info['units'])
            }

            return result

    def test_context_retrieval(self) -> Dict[str, Any]:
        """컨텍스트 검색 테스트"""
        from unittest.mock import patch

        # 임베딩 생성 Mock
        with patch.object(self.pipeline.embeddings_manager, 'generate_single_embedding') as mock_embed:
            mock_embed.return_value = [0.1] * 1536

            # 컨텍스트 검색
            contexts = self.pipeline.retriever.retrieve_context(
                query="일차함수의 기울기",
                subject="수학",
                unit="일차함수",
                k=2
            )

            result = {
                'contexts_found': len(contexts),
                'relevant_content': any('일차함수' in context for context in contexts),
                'search_worked': len(contexts) > 0
            }

            return result

    def test_question_generation(self) -> Dict[str, Any]:
        """문제 생성 테스트"""
        from unittest.mock import patch, MagicMock

        # LLM 응답 Mock
        mock_response = {
            "question": "일차함수 y = 2x + 3에서 기울기는 무엇인가?",
            "options": ["1", "2", "3", "4", "5"],
            "correct_answer": 2,
            "explanation": "일차함수 y = ax + b에서 a가 기울기이므로 답은 2입니다.",
            "difficulty": "medium",
            "subject": "수학",
            "unit": "일차함수"
        }

        with patch.object(self.pipeline.llm_client, 'generate_structured_response') as mock_llm, \
             patch.object(self.pipeline.embeddings_manager, 'generate_single_embedding') as mock_embed:

            mock_llm.return_value = mock_response
            mock_embed.return_value = [0.1] * 1536

            # 문제 생성
            question = self.pipeline.question_generator.generate_question(
                subject="수학",
                unit="일차함수",
                difficulty="medium"
            )

            # 검증
            is_valid = self.pipeline.question_generator.validate_question(question)

            result = {
                'question_generated': question is not None,
                'has_correct_format': is_valid,
                'has_5_options': len(question.get('options', [])) == 5,
                'has_explanation': bool(question.get('explanation', '')),
                'correct_answer_valid': 1 <= question.get('correct_answer', 0) <= 5
            }

            # 저장
            self.test_question = question

            return result

    def test_batch_processing(self) -> Dict[str, Any]:
        """배치 처리 테스트"""
        from unittest.mock import patch

        mock_response = {
            "question": "테스트 문제",
            "options": ["1", "2", "3", "4", "5"],
            "correct_answer": 1,
            "explanation": "테스트 해설",
            "difficulty": "medium",
            "subject": "수학",
            "unit": "일차함수"
        }

        with patch.object(self.pipeline.llm_client, 'generate_structured_response') as mock_llm, \
             patch.object(self.pipeline.embeddings_manager, 'generate_single_embedding') as mock_embed:

            mock_llm.return_value = mock_response
            mock_embed.return_value = [0.1] * 1536

            # 배치 문제 생성
            questions = self.pipeline.question_generator.generate_batch_questions(
                subject="수학",
                unit="일차함수",
                count=3,
                difficulty="medium"
            )

            result = {
                'batch_count': len(questions),
                'all_valid': all(self.pipeline.question_generator.validate_question(q) for q in questions),
                'unique_questions': len(set(q['question'] for q in questions)) == len(questions)
            }

            return result

    def test_performance(self) -> Dict[str, Any]:
        """성능 테스트"""
        from unittest.mock import patch

        # 많은 문서로 성능 테스트
        large_documents = []
        for i in range(20):
            from src.rag.document_processor import Document
            large_documents.append(Document(
                content=f"테스트 문서 {i}의 내용입니다. " * 10,
                metadata={"subject": "수학", "unit": f"단원{i}", "index": i}
            ))

        # 저장 성능 측정
        with patch.object(self.pipeline.embeddings_manager, 'generate_embeddings') as mock_embed:
            mock_embed.return_value = [[0.1 + i * 0.01] * 1536 for i in range(len(large_documents))]

            start_time = time.time()
            embeddings = mock_embed.return_value
            success = self.pipeline.vector_store.add_documents(large_documents, embeddings)
            storage_time = time.time() - start_time

        # 검색 성능 측정
        with patch.object(self.pipeline.embeddings_manager, 'generate_single_embedding') as mock_embed:
            mock_embed.return_value = [0.1] * 1536

            start_time = time.time()
            contexts = self.pipeline.retriever.retrieve_context("테스트 쿼리", k=5)
            search_time = time.time() - start_time

        result = {
            'storage_time': storage_time,
            'search_time': search_time,
            'documents_stored': len(large_documents),
            'storage_success': success,
            'search_results': len(contexts),
            'performance_acceptable': storage_time < 5.0 and search_time < 1.0
        }

        return result

    def test_error_handling(self) -> Dict[str, Any]:
        """에러 처리 테스트"""
        from unittest.mock import patch

        result = {}

        # 1. 잘못된 파일 경로
        try:
            self.pipeline.document_processor.load_textbook("nonexistent.txt", "수학", "일차함수")
            result['file_error_handling'] = False
        except Exception:
            result['file_error_handling'] = True

        # 2. 빈 컨텍스트 처리
        with patch.object(self.pipeline.retriever, 'retrieve_context', return_value=[]):
            try:
                self.pipeline.question_generator.generate_question("수학", "일차함수", "medium")
                result['empty_context_handling'] = False
            except Exception:
                result['empty_context_handling'] = True

        # 3. 잘못된 문제 형식 검증
        invalid_question = {
            "question": "",  # 빈 문제
            "options": ["1", "2"],  # 부족한 선택지
            "correct_answer": 6,  # 잘못된 정답
        }
        result['question_validation'] = not self.pipeline.question_generator.validate_question(invalid_question)

        return result

    def _print_test_summary(self):
        """테스트 결과 요약 출력"""
        print("\n" + "="*60)
        print("🧪 파이프라인 테스트 결과")
        print("="*60)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'success')
        failed_tests = total_tests - passed_tests
        total_time = sum(result['duration'] for result in self.test_results.values())

        print(f"\n📊 전체 결과:")
        print(f"   총 테스트: {total_tests}")
        print(f"   성공: {passed_tests}")
        print(f"   실패: {failed_tests}")
        print(f"   총 소요시간: {total_time:.2f}초")

        print(f"\n📋 테스트 상세:")
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            print(f"   {status_icon} {test_name} ({result['duration']:.2f}초)")

            if result['status'] == 'failed':
                print(f"      오류: {result['error']}")

        if failed_tests == 0:
            print(f"\n🎉 모든 테스트가 성공했습니다!")
        else:
            print(f"\n⚠️  {failed_tests}개의 테스트가 실패했습니다.")

        print("\n💡 추천 사항:")
        if failed_tests > 0:
            print("   - 실패한 테스트의 오류 메시지를 확인하세요")
            print("   - OpenAI API 키가 올바르게 설정되었는지 확인하세요")
            print("   - 의존성 패키지가 모두 설치되었는지 확인하세요")
        else:
            print("   - 실제 데이터로 시스템을 테스트해보세요")
            print("   - 프로덕션 환경에서 성능을 모니터링하세요")

        print("\n" + "="*60)


def main():
    """메인 함수"""
    tester = PipelineTester()

    try:
        results = tester.run_all_tests()

        # 결과를 JSON 파일로 저장
        results_file = Path(tester.project_root) / "test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n📄 상세 결과가 {results_file}에 저장되었습니다.")

        # 성공/실패 여부 반환
        failed_count = sum(1 for result in results.values() if result['status'] == 'failed')
        return 0 if failed_count == 0 else 1

    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())