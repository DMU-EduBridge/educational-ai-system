#!/usr/bin/env python3
"""
환경 설정 스크립트
Educational AI System 초기 설정을 수행합니다.
"""

import os
import sys
from pathlib import Path
import subprocess
import shutil
from typing import List, Tuple

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_application_logger


class EnvironmentSetup:
    """환경 설정 클래스"""

    def __init__(self):
        self.logger = setup_application_logger("setup")
        self.project_root = Path(__file__).parent.parent
        self.errors = []
        self.warnings = []

    def run_setup(self) -> bool:
        """전체 설정 프로세스 실행"""
        self.logger.info("Educational AI System 환경 설정을 시작합니다...")

        try:
            # 1. Python 버전 확인
            self.check_python_version()

            # 2. 필요한 디렉토리 생성
            self.create_directories()

            # 3. .env 파일 생성
            self.setup_env_file()

            # 4. 의존성 확인
            self.check_dependencies()

            # 5. ChromaDB 초기화 테스트
            self.test_chromadb()

            # 6. 샘플 데이터 확인
            self.verify_sample_data()

            # 7. 권한 설정
            self.set_permissions()

            # 결과 출력
            self.print_summary()

            return len(self.errors) == 0

        except Exception as e:
            self.logger.error(f"설정 중 오류 발생: {str(e)}")
            self.errors.append(f"Setup failed: {str(e)}")
            return False

    def check_python_version(self):
        """Python 버전 확인"""
        self.logger.info("Python 버전 확인 중...")

        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            error_msg = f"Python 3.8 이상이 필요합니다. 현재 버전: {version.major}.{version.minor}"
            self.errors.append(error_msg)
            self.logger.error(error_msg)
        else:
            self.logger.info(f"Python 버전 확인 완료: {version.major}.{version.minor}.{version.micro}")

    def create_directories(self):
        """필요한 디렉토리 생성"""
        self.logger.info("필요한 디렉토리 생성 중...")

        directories = [
            "data/vector_db",
            "data/cache",
            "data/sample_textbooks",
            "logs",
            "test_data/vector_db",
            "test_data/cache"
        ]

        for dir_path in directories:
            full_path = self.project_root / dir_path
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"디렉토리 생성: {full_path}")
            except Exception as e:
                error_msg = f"디렉토리 생성 실패 {full_path}: {str(e)}"
                self.errors.append(error_msg)
                self.logger.error(error_msg)

    def setup_env_file(self):
        """환경 설정 파일 생성"""
        self.logger.info("환경 설정 파일 확인 중...")

        env_file = self.project_root / ".env"
        env_example = self.project_root / ".env.example"

        if not env_file.exists():
            if env_example.exists():
                try:
                    shutil.copy2(env_example, env_file)
                    self.logger.info(".env 파일이 생성되었습니다.")
                    self.warnings.append("⚠️  .env 파일에서 OPENAI_API_KEY를 설정해주세요!")
                except Exception as e:
                    error_msg = f".env 파일 생성 실패: {str(e)}"
                    self.errors.append(error_msg)
                    self.logger.error(error_msg)
            else:
                error_msg = ".env.example 파일이 없습니다."
                self.errors.append(error_msg)
                self.logger.error(error_msg)
        else:
            self.logger.info(".env 파일이 이미 존재합니다.")

        # OpenAI API 키 확인
        if env_file.exists():
            self.check_openai_api_key(env_file)

    def check_openai_api_key(self, env_file: Path):
        """OpenAI API 키 확인"""
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                content = f.read()

            if "OPENAI_API_KEY=your_openai_api_key_here" in content:
                self.warnings.append("⚠️  OpenAI API 키가 기본값으로 설정되어 있습니다. 실제 키로 변경해주세요!")
            elif "OPENAI_API_KEY=" in content:
                # 간단한 형식 검사
                for line in content.split('\n'):
                    if line.startswith('OPENAI_API_KEY='):
                        key = line.split('=', 1)[1].strip()
                        if key and key.startswith('sk-') and len(key) > 20:
                            self.logger.info("OpenAI API 키가 올바른 형식으로 설정되어 있습니다.")
                        elif key:
                            self.warnings.append("⚠️  OpenAI API 키 형식을 확인해주세요!")
                        break
            else:
                self.warnings.append("⚠️  .env 파일에 OPENAI_API_KEY가 설정되지 않았습니다!")

        except Exception as e:
            self.warnings.append(f"⚠️  .env 파일 읽기 실패: {str(e)}")

    def check_dependencies(self):
        """의존성 패키지 확인"""
        self.logger.info("의존성 패키지 확인 중...")

        required_packages = [
            'openai',
            'chromadb',
            'click',
            'pydantic',
            'tiktoken',
            'numpy',
            'pytest'
        ]

        missing_packages = []

        for package in required_packages:
            try:
                __import__(package)
                self.logger.info(f"✓ {package}")
            except ImportError:
                missing_packages.append(package)
                self.logger.warning(f"✗ {package} (누락)")

        if missing_packages:
            self.warnings.append(
                f"⚠️  누락된 패키지가 있습니다: {', '.join(missing_packages)}\n"
                f"   다음 명령어로 설치하세요: pip install -r requirements.txt"
            )

    def test_chromadb(self):
        """ChromaDB 초기화 테스트"""
        self.logger.info("ChromaDB 초기화 테스트 중...")

        try:
            import chromadb
            from chromadb.config import Settings

            # 테스트용 클라이언트 생성
            test_db_path = self.project_root / "test_data" / "setup_test_db"
            test_db_path.mkdir(parents=True, exist_ok=True)

            client = chromadb.PersistentClient(
                path=str(test_db_path),
                settings=Settings(anonymized_telemetry=False)
            )

            # 테스트 컬렉션 생성
            collection = client.create_collection(
                name="setup_test",
                metadata={"description": "Setup test collection"}
            )

            # 간단한 데이터 추가 테스트
            collection.add(
                ids=["test1"],
                embeddings=[[0.1] * 1536],
                documents=["테스트 문서"],
                metadatas=[{"test": True}]
            )

            # 검색 테스트
            results = collection.query(
                query_embeddings=[[0.1] * 1536],
                n_results=1
            )

            if results['documents'] and results['documents'][0]:
                self.logger.info("ChromaDB 테스트 성공")
            else:
                self.warnings.append("⚠️  ChromaDB 검색 테스트 실패")

            # 테스트 데이터 정리
            client.delete_collection("setup_test")

            # 테스트 디렉토리 정리
            if test_db_path.exists():
                shutil.rmtree(test_db_path, ignore_errors=True)

        except Exception as e:
            error_msg = f"ChromaDB 테스트 실패: {str(e)}"
            self.errors.append(error_msg)
            self.logger.error(error_msg)

    def verify_sample_data(self):
        """샘플 데이터 확인"""
        self.logger.info("샘플 데이터 확인 중...")

        sample_files = [
            "data/sample_textbooks/math_unit1.txt",
            "data/sample_textbooks/science_unit1.txt"
        ]

        for file_path in sample_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                # 파일 크기 확인
                size = full_path.stat().st_size
                if size > 100:  # 최소 100바이트
                    self.logger.info(f"✓ {file_path} ({size} bytes)")
                else:
                    self.warnings.append(f"⚠️  {file_path} 파일이 너무 작습니다 ({size} bytes)")
            else:
                self.warnings.append(f"⚠️  샘플 파일이 없습니다: {file_path}")

    def set_permissions(self):
        """파일 권한 설정 (Unix 시스템만)"""
        if os.name != 'posix':
            return

        self.logger.info("파일 권한 설정 중...")

        try:
            # main.py 실행 권한 추가
            main_py = self.project_root / "src" / "main.py"
            if main_py.exists():
                os.chmod(main_py, 0o755)

            # 스크립트 파일들 실행 권한 추가
            scripts_dir = self.project_root / "scripts"
            for script_file in scripts_dir.glob("*.py"):
                os.chmod(script_file, 0o755)

            self.logger.info("파일 권한 설정 완료")

        except Exception as e:
            self.warnings.append(f"⚠️  권한 설정 실패: {str(e)}")

    def print_summary(self):
        """설정 결과 요약 출력"""
        print("\n" + "="*60)
        print("🎓 Educational AI System 환경 설정 완료")
        print("="*60)

        if not self.errors and not self.warnings:
            print("✅ 모든 설정이 성공적으로 완료되었습니다!")
        else:
            if self.errors:
                print("\n❌ 오류:")
                for error in self.errors:
                    print(f"   {error}")

            if self.warnings:
                print("\n⚠️  주의사항:")
                for warning in self.warnings:
                    print(f"   {warning}")

        print("\n📋 다음 단계:")
        print("1. .env 파일에서 OpenAI API 키 설정")
        print("2. 의존성 패키지 설치: pip install -r requirements.txt")
        print("3. 파이프라인 테스트: python -m src.main test-pipeline")
        print("4. 샘플 교과서 처리: python -m src.main process-textbook --file data/sample_textbooks/math_unit1.txt --subject 수학 --unit 일차함수")

        print("\n🔗 유용한 명령어:")
        print("   python -m src.main --help          # 도움말")
        print("   python -m src.main status          # 시스템 상태")
        print("   pytest tests/ -v                   # 테스트 실행")

        print("\n💡 문제 해결:")
        print("   - OpenAI API 키 관련: https://platform.openai.com/api-keys")
        print("   - 설치 문제: README.md 참조")
        print("   - GitHub Issues: 문제 보고 및 질문")

        print("\n" + "="*60)


def main():
    """메인 함수"""
    setup = EnvironmentSetup()
    success = setup.run_setup()

    if success:
        print("🎉 설정이 성공적으로 완료되었습니다!")
        return 0
    else:
        print("❌ 설정 중 오류가 발생했습니다. 위의 오류 메시지를 확인해주세요.")
        return 1


if __name__ == "__main__":
    sys.exit(main())