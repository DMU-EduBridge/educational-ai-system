#!/usr/bin/env python3
"""
RAG 기반 5지선다 문제 10개 생성 시스템
메인 CLI 애플리케이션
"""

import click
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_settings, check_environment, setup_environment
from document_loader import DocumentLoader
from rag_processor import RAGProcessor
from question_generator import QuestionGenerator


def print_banner():
    """애플리케이션 배너 출력"""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                     RAG 기반 5지선다 문제 생성 시스템                              ║
║                    Retrieval-Augmented Generation MCQ Generator               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)


@click.command()
@click.option('--input-file', '-i',
              type=click.Path(exists=True),
              help='입력 텍스트 문서 경로')
@click.option('--output-file', '-o',
              type=click.Path(),
              help='출력 JSON 파일 경로 (기본값: data/output/questions_YYYYMMDD_HHMMSS.json)')
@click.option('--num-questions', '-n',
              default=10,
              type=click.IntRange(1, 20),
              help='생성할 문제 수 (1-20, 기본값: 10)')
@click.option('--difficulty-mix', '-d',
              default='balanced',
              type=click.Choice(['easy', 'medium', 'hard', 'balanced']),
              help='난이도 구성 (기본값: balanced)')
@click.option('--check-env',
              is_flag=True,
              help='환경 설정 상태만 확인')
@click.option('--setup-env',
              is_flag=True,
              help='환경 설정 초기화')
@click.option('--verbose', '-v',
              is_flag=True,
              help='상세 출력')
def generate_questions(input_file, output_file, num_questions, difficulty_mix,
                      check_env, setup_env, verbose):
    """
    RAG 기반 5지선다 문제 생성

    사용 예시:
    python src/main.py -i data/input/document.txt -o data/output/questions.json -n 10
    """

    print_banner()

    # 환경 설정 확인만 하는 경우
    if check_env:
        print("🔍 환경 설정 상태 확인 중...")
        status = check_environment()

        print("\n📊 환경 설정 상태:")
        print(f"  .env 파일: {'✅' if status['env_file_exists'] else '❌'}")
        print(f"  API 키 설정: {'✅' if status['api_key_set'] else '❌'}")
        print(f"  API 키 유효성: {'✅' if status['api_key_valid'] else '❌'}")
        print(f"  벡터 DB 경로: {'✅' if status['vector_db_path_exists'] else '❌'}")

        if status["issues"]:
            print("\n⚠️ 발견된 문제:")
            for issue in status["issues"]:
                print(f"  - {issue}")
        else:
            print("\n✅ 모든 환경 설정이 정상입니다!")

        return

    # 환경 설정 초기화
    if setup_env:
        setup_environment()
        return

    # 입력 파일 검증 (실제 처리가 필요한 경우에만)
    if not input_file:
        click.echo("❌ 오류: 입력 파일이 필요합니다. --input-file 옵션을 사용하세요.", err=True)
        click.echo("💡 도움말: python src/main.py --help", err=True)
        sys.exit(1)

    try:
        # 1. 환경 설정 로드
        if verbose:
            print("⚙️ 환경 설정 로드 중...")

        try:
            settings = get_settings()
        except Exception as e:
            click.echo(f"❌ 환경 설정 오류: {str(e)}", err=True)
            click.echo("💡 해결 방법: python src/main.py --setup-env 실행", err=True)
            sys.exit(1)

        # 2. 문서 로딩
        print("📄 문서 로딩 중...")
        try:
            document_text = DocumentLoader.load_and_process(input_file)
        except Exception as e:
            click.echo(f"❌ 문서 로딩 실패: {str(e)}", err=True)
            sys.exit(1)

        # 3. RAG 처리
        print("🔍 RAG 처리 중...")
        try:
            rag_processor = RAGProcessor(
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap
            )

            success = rag_processor.process_document(document_text)
            if not success:
                raise Exception("RAG 처리에 실패했습니다")

            # 처리 결과 출력
            collection_info = rag_processor.get_collection_info()
            if verbose:
                print(f"  📊 처리 결과: {collection_info['total_chunks']}개 청크 생성")

        except Exception as e:
            click.echo(f"❌ RAG 처리 실패: {str(e)}", err=True)
            sys.exit(1)

        # 4. 문제 생성
        print(f"❓ {num_questions}개 문제 생성 중...")
        try:
            question_generator = QuestionGenerator(
                rag_processor=rag_processor,
                llm_model=settings.openai_model
            )

            questions = question_generator.generate_questions(
                num_questions=num_questions,
                difficulty_mix=difficulty_mix
            )

            if not questions:
                raise Exception("문제 생성에 실패했습니다")

        except Exception as e:
            click.echo(f"❌ 문제 생성 실패: {str(e)}", err=True)
            sys.exit(1)

        # 5. 결과 구성
        print("📊 결과 구성 중...")

        # 난이도별 통계
        difficulty_stats = {}
        for question in questions:
            diff = question.get('difficulty', 'unknown')
            difficulty_stats[diff] = difficulty_stats.get(diff, 0) + 1

        # 최종 결과 구성
        result = {
            "metadata": {
                "source_document": Path(input_file).name,
                "generated_at": datetime.now().isoformat(),
                "total_questions": len(questions),
                "difficulty_distribution": difficulty_stats,
                "settings": {
                    "model": settings.openai_model,
                    "chunk_size": settings.chunk_size,
                    "difficulty_mix": difficulty_mix
                }
            },
            "questions": questions
        }

        # 6. 파일 저장
        if not output_file:
            # 기본 출력 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"../data/output/questions_{timestamp}.json"

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print("💾 결과 저장 중...")

        # 7. 결과 출력
        print("\n" + "="*80)
        print("✅ 문제 생성 완료!")
        print("="*80)
        print(f"📄 입력 파일: {Path(input_file).name}")
        print(f"📝 생성된 문제: {len(questions)}개")
        print(f"🎯 난이도 분포: {difficulty_stats}")
        print(f"💾 출력 파일: {output_path}")
        print(f"📊 파일 크기: {output_path.stat().st_size / 1024:.1f}KB")

        if verbose:
            print("\n📋 생성된 문제 미리보기:")
            print("-" * 80)
            for i, question in enumerate(questions[:3], 1):  # 처음 3개 문제만 표시
                print(f"\n[문제 {i}] ({question['difficulty']}) {question['type']}")
                print(f"Q: {question['question']}")
                print("선택지:")
                for j, option in enumerate(question['options'], 1):
                    marker = "✓" if j == question['correct_answer'] else " "
                    print(f"  {marker} {j}. {option}")
                print(f"해설: {question['explanation']}")

            if len(questions) > 3:
                print(f"\n... 및 {len(questions) - 3}개 문제 더")

        print(f"\n🎉 모든 작업이 완료되었습니다!")
        print(f"📂 결과 파일: {output_path}")

    except KeyboardInterrupt:
        print("\n❌ 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ 예상치 못한 오류: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    generate_questions()