#!/usr/bin/env python3
"""
Setup Script - RAG 기반 문제 생성 시스템 설정
"""

import os
import sys
from pathlib import Path


def create_directories():
    """필요한 디렉토리 생성"""
    directories = [
        "data/input",
        "data/output",
        "data/vector_db",
        "logs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 디렉토리 생성: {directory}")


def check_requirements():
    """필요한 라이브러리 확인"""
    required_packages = [
        "openai",
        "chromadb",
        "langchain",
        "click",
        "pydantic",
        "python-dotenv"
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ 누락된 패키지:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n설치 명령어: pip install -r requirements.txt")
        return False
    else:
        print("✅ 모든 필수 패키지가 설치되어 있습니다")
        return True


def setup_environment():
    """환경 설정"""
    env_file = Path(".env")
    env_example = Path(".env.example")

    if not env_file.exists() and env_example.exists():
        # .env.example을 .env로 복사
        with open(env_example, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        print("📝 .env 파일 생성됨 (.env.example에서 복사)")
        print("🔧 .env 파일에서 OPENAI_API_KEY를 설정하세요")
        return False
    elif env_file.exists():
        print("✅ .env 파일이 이미 존재합니다")
        return True
    else:
        print("❌ .env.example 파일이 없습니다")
        return False


def main():
    """메인 설정 함수"""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                     RAG 기반 문제 생성 시스템 설정                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

    print("🚀 시스템 설정을 시작합니다...\n")

    # 1. 디렉토리 생성
    print("1️⃣ 디렉토리 설정")
    create_directories()
    print()

    # 2. 패키지 확인
    print("2️⃣ 패키지 확인")
    packages_ok = check_requirements()
    print()

    # 3. 환경 설정
    print("3️⃣ 환경 설정")
    env_ok = setup_environment()
    print()

    # 4. 결과 출력
    print("📊 설정 결과:")
    print(f"  디렉토리: ✅")
    print(f"  패키지: {'✅' if packages_ok else '❌'}")
    print(f"  환경 설정: {'✅' if env_ok else '⚠️'}")

    if packages_ok and env_ok:
        print("\n🎉 설정이 완료되었습니다!")
        print("\n📋 다음 단계:")
        print("  1. .env 파일에서 OPENAI_API_KEY 설정")
        print("  2. 테스트 실행: python src/main.py --check-env")
        print("  3. 샘플 실행: python src/main.py -i examples/sample_document.txt")
    else:
        print("\n⚠️ 설정을 완료하려면 누락된 항목들을 해결하세요")


if __name__ == "__main__":
    main()