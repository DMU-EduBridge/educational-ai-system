#!/usr/bin/env python3
"""
Educational AI System - Main Entry Point

이 프로젝트는 중학교 교과서를 기반으로 자동으로 5지선다 문제를 생성하는 
RAG(Retrieval-Augmented Generation) 시스템입니다.

Usage:
    python main.py --help
    cd ai-services && python -m src.main --help
    
Examples:
    # 환경 설정
    cd ai-services && python scripts/setup_environment.py
    
    # 교과서 처리
    cd ai-services && python -m src.main process-textbook \\
        --file data/sample_textbooks/math_unit1.txt \\
        --subject 수학 --unit 일차함수
    
    # 문제 생성
    cd ai-services && python -m src.main generate-questions \\
        --subject 수학 --unit 일차함수 --difficulty medium --count 3
"""

import sys
from pathlib import Path


def main():
    """메인 엔트리 포인트"""
    print("🎓 Educational AI System")
    print("=" * 50)
    print(__doc__.strip())
    
    ai_services_path = Path(__file__).parent / "ai-services"
    if not ai_services_path.exists():
        print("❌ ai-services 디렉토리를 찾을 수 없습니다.")
        sys.exit(1)
    
    print(f"\n📁 AI Services 디렉토리: {ai_services_path}")
    print("\n💡 실제 애플리케이션을 실행하려면:")
    print("   cd ai-services")
    print("   python -m src.main --help")


if __name__ == "__main__":
    main()
