#!/usr/bin/env python3
"""
Educational AI System - RAG Pipeline
메인 CLI 애플리케이션
"""

import click
import json
import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import traceback

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 로컬 모듈 임포트
try:
    from src.utils.config import get_settings, Settings
    from src.utils.logger import setup_application_logger, get_logger
    from src.rag.document_processor import DocumentProcessor
    from src.rag.embeddings import EmbeddingsManager
    from src.rag.vector_store import VectorStore
    from src.rag.retriever import RAGRetriever
    from src.models.llm_client import LLMClient
    from src.models.question_generator import QuestionGenerator
    from src.evaluation.quality_assessor import QualityAssessor
except ImportError:
    # 패키지가 설치된 경우의 import
    from utils.config import get_settings, Settings
    from utils.logger import setup_application_logger, get_logger
    from rag.document_processor import DocumentProcessor
    from rag.embeddings import EmbeddingsManager
    from rag.vector_store import VectorStore
    from rag.retriever import RAGRetriever
    from models.llm_client import LLMClient
    from models.question_generator import QuestionGenerator
    from evaluation.quality_assessor import QualityAssessor


class RAGPipeline:
    """RAG 파이프라인 메인 클래스"""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.logger = get_logger(__name__)

        # 컴포넌트 초기화
        self._initialize_components()

    def _initialize_components(self):
        """RAG 파이프라인 컴포넌트 초기화"""
        try:
            # API 키 검증
            if not self.settings.validate_api_key():
                raise ValueError("Invalid or missing OpenAI API key")

            # 각 컴포넌트 초기화
            self.document_processor = DocumentProcessor()

            self.embeddings_manager = EmbeddingsManager(
                model_name=self.settings.openai_embedding_model,
                api_key=self.settings.openai_api_key
            )

            self.vector_store = VectorStore(
                collection_name=self.settings.chroma_collection_name,
                persist_directory=self.settings.chroma_db_path
            )

            self.retriever = RAGRetriever(
                vector_store=self.vector_store,
                embeddings_manager=self.embeddings_manager
            )

            self.llm_client = LLMClient(
                model_name=self.settings.openai_model,
                api_key=self.settings.openai_api_key,
                temperature=self.settings.openai_temperature,
                max_tokens=self.settings.openai_max_tokens
            )

            self.question_generator = QuestionGenerator(
                llm_client=self.llm_client,
                retriever=self.retriever
            )

            self.quality_assessor = QualityAssessor(llm_client=self.llm_client)

            self.logger.info("RAG Pipeline initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize RAG Pipeline: {str(e)}")
            raise

    def process_textbook(self, file_path: str, subject: str, unit: str) -> Dict[str, Any]:
        """교과서 처리 및 벡터 DB 저장"""
        try:
            self.logger.info(f"Processing textbook: {file_path}")

            # 1. 문서 로드 및 처리
            documents = self.document_processor.load_textbook(file_path, subject, unit)
            self.logger.info(f"Loaded {len(documents)} document chunks")

            # 2. 임베딩 생성
            texts = [doc.content for doc in documents]
            cost_info = self.embeddings_manager.estimate_cost(texts)
            self.logger.info(f"Estimated embedding cost: ${cost_info['estimated_cost_usd']:.4f}")

            embeddings = self.embeddings_manager.generate_embeddings(texts)
            self.logger.info(f"Generated {len(embeddings)} embeddings")

            # 3. 벡터 저장소에 저장
            success = self.vector_store.add_documents(documents, embeddings)

            if success:
                result = {
                    'status': 'success',
                    'processed_chunks': len(documents),
                    'total_tokens': cost_info['total_tokens'],
                    'estimated_cost': cost_info['estimated_cost_usd'],
                    'subject': subject,
                    'unit': unit,
                    'source_file': Path(file_path).name
                }
                self.logger.info("Textbook processing completed successfully")
                return result
            else:
                raise Exception("Failed to store documents in vector database")

        except Exception as e:
            self.logger.error(f"Error processing textbook: {str(e)}")
            raise

    def generate_questions(self,
                         subject: str,
                         unit: str,
                         difficulty: str = 'medium',
                         count: int = 1) -> List[Dict[str, Any]]:
        """문제 생성"""
        try:
            self.logger.info(f"Generating {count} questions for {subject} - {unit} ({difficulty})")

            if count == 1:
                question = self.question_generator.generate_question(subject, unit, difficulty)
                return [question]
            else:
                questions = self.question_generator.generate_batch_questions(
                    subject, unit, count, difficulty
                )
                return questions

        except Exception as e:
            self.logger.error(f"Error generating questions: {str(e)}")
            raise

    def evaluate_questions(self, question_file: str, subject: str, unit: str) -> List[Dict[str, Any]]:
        """생성된 문제 품질 평가"""
        try:
            self.logger.info(f"Evaluating questions from {question_file}")
            with open(question_file, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)

            results = []
            for i, question_data in enumerate(questions_data):
                self.logger.debug(f"Evaluating question {i+1}")
                # Retrieve context based on the question text itself to find the most relevant source
                retrieved_docs = self.retriever.retrieve_documents(
                    query=question_data['question'],
                    subject=subject,
                    unit=unit
                )
                source_context = "\n\n".join([doc.content for doc in retrieved_docs])

                if not source_context:
                    self.logger.warning(f"Could not retrieve source context for question {i+1}. Skipping assessment.")
                    continue

                assessment = self.quality_assessor.assess_question(question_data, source_context)
                results.append({"question_id": i + 1, "assessment": assessment, "original_question": question_data})
            
            self.logger.info(f"Finished evaluating {len(results)} questions.")
            return results
        except Exception as e:
            self.logger.error(f"Error evaluating questions: {str(e)}")
            raise

    def get_status(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        try:
            # 벡터 DB 정보
            collection_info = self.vector_store.get_collection_info()

            # LLM 사용량
            llm_usage = self.llm_client.track_usage()

            # 문제 생성 통계
            question_stats = self.question_generator.get_question_statistics()

            status = {
                'vector_store': collection_info,
                'llm_usage': llm_usage,
                'question_generation': question_stats,
                'settings': self.settings.to_dict()
            }

            return status

        except Exception as e:
            self.logger.error(f"Error getting status: {str(e)}")
            raise

    def test_pipeline(self) -> Dict[str, Any]:
        """전체 파이프라인 테스트"""
        try:
            self.logger.info("Starting pipeline test")

            # 테스트 텍스트
            test_content = "일차함수는 y = ax + b 형태의 함수입니다. 여기서 a는 기울기, b는 y절편을 나타냅니다."

            # 임시 텍스트 파일 생성
            test_file = Path("./test_content.txt")
            test_file.write_text(test_content, encoding='utf-8')

            try:
                # 1. 문서 처리 테스트
                process_result = self.process_textbook(str(test_file), "수학", "일차함수")

                # 2. 문제 생성 테스트
                questions = self.generate_questions("수학", "일차함수", "medium", 1)

                # 3. 상태 확인
                status = self.get_status()

                test_result = {
                    'status': 'success',
                    'document_processing': process_result,
                    'question_generation': {
                        'generated_questions': len(questions),
                        'sample_question': questions[0] if questions else None
                    },
                    'system_status': status
                }

                self.logger.info("Pipeline test completed successfully")
                return test_result

            finally:
                # 테스트 파일 정리
                if test_file.exists():
                    test_file.unlink()

        except Exception as e:
            self.logger.error(f"Pipeline test failed: {str(e)}")
            raise


# CLI 애플리케이션 정의
@click.group()
@click.option('--config', type=click.Path(exists=True), help='설정 파일 경로')
@click.option('--debug', is_flag=True, help='디버그 모드')
@click.option('--verbose', is_flag=True, help='상세 출력')
@click.pass_context
def cli(ctx, config, debug, verbose):
    """Educational AI System - RAG Pipeline"""
    ctx.ensure_object(dict)

    # 설정 로드
    if config:
        # 설정 파일이 제공된 경우 (향후 구현)
        settings = get_settings()
    else:
        settings = get_settings()

    # 디버그/상세 모드 설정
    if debug:
        settings.debug = True
        settings.log_level = "DEBUG"
    if verbose:
        settings.verbose = True

    # 로거 설정
    logger = setup_application_logger(
        "educational_ai",
        config={
            'log_level': settings.log_level,
            'debug': settings.debug,
            'verbose': settings.verbose
        }
    )

    # 컨텍스트에 설정과 파이프라인 저장
    ctx.obj['settings'] = settings
    ctx.obj['logger'] = logger

    try:
        ctx.obj['pipeline'] = RAGPipeline(settings)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        if debug:
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--file', required=True, type=click.Path(exists=True), help='교과서 파일 경로')
@click.option('--subject', required=True, help='과목명')
@click.option('--unit', required=True, help='단원명')
@click.pass_context
def process_textbook(ctx, file, subject, unit):
    """교과서 처리 및 벡터 DB 저장"""
    pipeline = ctx.obj['pipeline']
    logger = ctx.obj['logger']

    try:
        result = pipeline.process_textbook(file, subject, unit)

        click.echo(f"✅ 교과서 처리 완료!")
        click.echo(f"   파일: {result['source_file']}")
        click.echo(f"   과목: {result['subject']}")
        click.echo(f"   단원: {result['unit']}")
        click.echo(f"   처리된 청크: {result['processed_chunks']}개")
        click.echo(f"   토큰 수: {result['total_tokens']:,}")
        click.echo(f"   예상 비용: ${result['estimated_cost']:.4f}")

    except Exception as e:
        click.echo(f"❌ 오류: {str(e)}", err=True)
        if ctx.obj['settings'].debug:
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--subject', required=True, help='과목명')
@click.option('--unit', required=True, help='단원명')
@click.option('--difficulty', default='medium', type=click.Choice(['easy', 'medium', 'hard']), help='난이도')
@click.option('--count', default=1, type=int, help='생성할 문제 수')
@click.option('--output', type=click.Path(), help='결과 저장 파일 (JSON)')
@click.pass_context
def generate_questions(ctx, subject, unit, difficulty, count, output):
    """문제 생성"""
    pipeline = ctx.obj['pipeline']
    logger = ctx.obj['logger']

    try:
        questions = pipeline.generate_questions(subject, unit, difficulty, count)

        click.echo(f"✅ 문제 생성 완료! ({len(questions)}개)")
        click.echo()

        for i, question in enumerate(questions, 1):
            click.echo(f"=== 문제 {i} ===")
            click.echo(f"난이도: {question['difficulty']}")
            click.echo(f"과목: {question['subject']} - {question['unit']}")
            click.echo()
            click.echo(f"문제: {question['question']}")
            click.echo()
            click.echo("선택지:")
            for j, option in enumerate(question['options'], 1):
                marker = "✓" if j == question['correct_answer'] else " "
                click.echo(f"  {marker} {j}. {option}")
            click.echo()
            click.echo(f"해설: {question['explanation']}")
            click.echo("-" * 50)

        # 파일 저장
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(questions, f, ensure_ascii=False, indent=2)

            click.echo(f"💾 결과가 {output}에 저장되었습니다.")

    except Exception as e:
        click.echo(f"❌ 오류: {str(e)}", err=True)
        if ctx.obj['settings'].debug:
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--question-file', required=True, type=click.Path(exists=True), help='평가할 문제 JSON 파일')
@click.option('--subject', required=True, help='과목명')
@click.option('--unit', required=True, help='단원명')
@click.option('--output', type=click.Path(), help='평가 결과 저장 파일 (JSON)')
@click.pass_context
def evaluate_questions(ctx, question_file, subject, unit, output):
    """생성된 문제의 품질을 평가합니다."""
    pipeline = ctx.obj['pipeline']
    logger = ctx.obj['logger']

    try:
        results = pipeline.evaluate_questions(question_file, subject, unit)
        click.echo(f"✅ 문제 품질 평가 완료! ({len(results)}개)")
        click.echo("=" * 50)

        for result in results:
            assessment = result['assessment']
            if 'error' in assessment:
                click.echo(f"ID {result['question_id']} 평가 실패: {assessment['error']}", err=True)
                continue

            click.echo(f"### 문제 ID: {result['question_id']} ###")
            click.echo(f"  질문: {result['original_question']['question'][:50]}...")
            click.echo(f"  사용 가능성: {'👍 Yes' if assessment.get('is_usable') else '👎 No'}")
            click.echo(f"  종합 점수: {assessment.get('overall_score', 'N/A')}")
            click.echo("  세부 점수:")
            for criterion, values in assessment.get('scores', {}).items():
                click.echo(f"    - {criterion.capitalize()}: {values.get('score')}/5")
            click.echo(f"  요약: {assessment.get('summary', 'N/A')}")
            click.echo("-" * 50)

        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            click.echo(f"💾 평가 결과가 {output}에 저장되었습니다.")

    except Exception as e:
        click.echo(f"❌ 오류: {str(e)}", err=True)
        if ctx.obj['settings'].debug:
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """벡터 DB 상태 확인"""
    pipeline = ctx.obj['pipeline']
    logger = ctx.obj['logger']

    try:
        status_info = pipeline.get_status()

        click.echo("📊 시스템 상태")
        click.echo("=" * 50)

        # 벡터 저장소 정보
        vs_info = status_info['vector_store']
        click.echo(f"🗄️  벡터 저장소:")
        click.echo(f"   컬렉션: {vs_info['collection_name']}")
        click.echo(f"   총 문서: {vs_info['total_documents']:,}개")
        click.echo(f"   과목: {', '.join(vs_info['subjects']) if vs_info['subjects'] else '없음'}")
        click.echo(f"   단원: {', '.join(vs_info['units']) if vs_info['units'] else '없음'}")
        click.echo()

        # LLM 사용량
        llm_info = status_info['llm_usage']
        click.echo(f"🤖 LLM 사용량:")
        click.echo(f"   모델: {llm_info['model']}")
        click.echo(f"   총 요청: {llm_info['total_requests']:,}회")
        click.echo(f"   총 토큰: {llm_info['total_tokens']:,}개")
        click.echo(f"   총 비용: ${llm_info['total_cost_usd']:.4f}")
        click.echo()

        # 문제 생성 통계
        qg_info = status_info['question_generation']
        click.echo(f"📝 문제 생성 통계:")
        click.echo(f"   생성된 문제: {qg_info['total_questions']}개")
        if qg_info['by_difficulty']:
            click.echo(f"   난이도별: {qg_info['by_difficulty']}")

    except Exception as e:
        click.echo(f"❌ 오류: {str(e)}", err=True)
        if ctx.obj['settings'].debug:
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.pass_context
def test_pipeline(ctx):
    """전체 파이프라인 테스트"""
    pipeline = ctx.obj['pipeline']
    logger = ctx.obj['logger']

    try:
        click.echo("🧪 파이프라인 테스트 시작...")

        result = pipeline.test_pipeline()

        click.echo("✅ 파이프라인 테스트 완료!")
        click.echo()
        click.echo("📊 테스트 결과:")
        click.echo(f"   문서 처리: {result['document_processing']['processed_chunks']}개 청크")
        click.echo(f"   문제 생성: {result['question_generation']['generated_questions']}개")
        click.echo()

        if result['question_generation']['sample_question']:
            sample = result['question_generation']['sample_question']
            click.echo("📝 생성된 샘플 문제:")
            click.echo(f"   {sample['question']}")

        click.echo()
        click.echo("✨ 모든 컴포넌트가 정상 작동합니다!")

    except Exception as e:
        click.echo(f"❌ 테스트 실패: {str(e)}", err=True)
        if ctx.obj['settings'].debug:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    cli()