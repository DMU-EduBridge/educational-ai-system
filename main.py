#!/usr/bin/env python3
"""
Educational AI System - Main Entry Point

이 프로젝트는 중학교 교과서를 기반으로 자동으로 5지선다 문제를 생성하는 
RAG(Retrieval-Augmented Generation) 시스템입니다.

Usage:
    python main.py --help
    python main.py [command] [options]
    
Examples:
    # 환경 설정 확인
    python main.py setup-env
    
    # 교과서 처리
    python main.py process-textbook \\
        --file ai-services/data/sample_textbooks/math_unit1.txt \\
        --subject 수학 --unit 일차함수
    
    # 문제 생성
    python main.py generate-questions \\
        --subject 수학 --unit 일차함수 --difficulty medium --count 3
"""

import sys
import os
import click
from pathlib import Path

# AI Services 모듈 경로 추가
current_dir = Path(__file__).parent
ai_services_dir = current_dir / "ai-services"
src_dir = ai_services_dir / "src"

if str(ai_services_dir) not in sys.path:
    sys.path.insert(0, str(ai_services_dir))

# AI Services CLI 가져오기 (선택적)
ai_services_cli = None
try:
    if (ai_services_dir / "src" / "main.py").exists():
        from src.main import cli as ai_services_cli
except ImportError:
    # AI Services CLI를 가져올 수 없는 경우 무시
    pass


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Educational AI System - RAG 기반 5지선다 문제 생성 시스템"""
    pass


@cli.command()
def info():
    """시스템 정보 출력"""
    print("🎓 Educational AI System v0.1.0")
    print("=" * 50)
    print(__doc__.strip())
    
    print(f"\n📁 프로젝트 경로: {current_dir}")
    print(f"📁 AI Services 경로: {ai_services_dir}")
    
    if ai_services_dir.exists():
        print("✅ AI Services 디렉토리가 존재합니다.")
    else:
        print("❌ AI Services 디렉토리를 찾을 수 없습니다.")
    
    print("\n💡 사용 가능한 명령어:")
    print("   python main.py info           - 시스템 정보")
    print("   python main.py setup-env      - 환경 설정")
    print("   python main.py ai-services    - AI Services CLI")


@cli.command()
def setup_env():
    """환경 설정 초기화"""
    try:
        setup_script = ai_services_dir / "scripts" / "setup_environment.py"
        if setup_script.exists():
            exec(open(setup_script).read())
        else:
            print("❌ 설정 스크립트를 찾을 수 없습니다.")
    except Exception as e:
        print(f"❌ 환경 설정 중 오류 발생: {e}")


@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
def ai_services(ctx):
    """AI Services CLI 실행"""
    import subprocess
    
    # ai-services의 main.py를 직접 실행
    ai_services_main = ai_services_dir / "src" / "main.py"
    
    if ai_services_main.exists():
        try:
            # uv run을 사용하여 ai-services CLI 실행
            cmd = ["uv", "run", "python", str(ai_services_main)] + ctx.args
            result = subprocess.run(cmd, cwd=current_dir)
            sys.exit(result.returncode)
        except Exception as e:
            print(f"❌ AI Services 실행 중 오류 발생: {e}")
            sys.exit(1)
    else:
        print("❌ AI Services CLI를 사용할 수 없습니다.")
        print("💡 다음 명령으로 직접 실행해보세요:")
        print("   cd ai-services")
        print("   uv run python src/main.py --help")
        sys.exit(1)


def main():
    """메인 엔트리 포인트"""
    cli()


if __name__ == "__main__":
    main()
