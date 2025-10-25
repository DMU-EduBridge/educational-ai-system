#!/usr/bin/env python3
"""
3학년 교과서 PDF 파일을 OCR 처리하여 벡터 DB에 저장하는 스크립트
"""

import sys
import os
from pathlib import Path
import json
from typing import List, Dict, Any
import time
from dotenv import load_dotenv

# 프로젝트 루트 경로 설정
script_dir = Path(__file__).resolve().parent
ai_services_dir = script_dir.parent
project_root = ai_services_dir.parent

# .env 파일 로드 (프로젝트 루트에서)
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded .env from {env_path}")
else:
    print(f"⚠️  .env file not found at {env_path}")

sys.path.insert(0, str(ai_services_dir))

from src.main import RAGPipeline
from src.rag.document_processor import DocumentProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


# 교과서별 단원 매핑
TEXTBOOK_UNITS = {
    "미래엔 중3 수학 교과서": {
        "01 실수와 그 계산.pdf": "실수와 그 계산",
        "02 이차방정식.pdf": "이차방정식",
        "03 이차함수.pdf": "이차함수",
        "04 삼각비.pdf": "삼각비",
        "05 원의 성질.pdf": "원의 성질",
        "06 통계.pdf": "통계",
    },
    "교학사(고) 중3 교과서 지도서": {
        # 하위 폴더 내 PDF 파일들 처리
    },
    "천재(이) 중3 수학 교과서": {
        # 하위 폴더 내 PDF 파일들 처리
    }
}

# 단일 PDF 파일 매핑
SINGLE_PDF_UNITS = {
    "비상 중3 수학 교과서.pdf": "통합교과서",
    "동아출판 (강옥기) 중3 수학 교과서.pdf": "통합교과서",
    "천재교육(류) 중3 수학 교과서.pdf": "통합교과서",
    "신사고 중3 교과서 4단원-삼각비.pdf": "삼각비",
    "[비상교육] 중등_수학 3_1_지도서 (1).pdf": "통합교과서",
}


def find_pdf_files(base_dir: Path) -> List[Dict[str, str]]:
    """
    3rd_grade_textbook 디렉토리에서 모든 PDF 파일을 찾아 정보 반환
    
    Returns:
        List of dicts with keys: filepath, subject, unit, publisher
    """
    pdf_files = []
    
    # 단일 PDF 파일들
    for filename, unit in SINGLE_PDF_UNITS.items():
        pdf_path = base_dir / filename
        if pdf_path.exists():
            # 출판사 추출
            publisher = filename.split()[0] if ' ' in filename else "미분류"
            
            pdf_files.append({
                'filepath': str(pdf_path),
                'subject': '수학',
                'unit': unit,
                'publisher': publisher,
                'filename': filename
            })
    
    # 폴더별 PDF 파일들
    for folder_name, unit_mapping in TEXTBOOK_UNITS.items():
        folder_path = base_dir / folder_name
        if not folder_path.exists():
            continue
        
        # 출판사 추출
        publisher = folder_name.split()[0] if ' ' in folder_name else folder_name
        
        for pdf_file in folder_path.glob("*.pdf"):
            # 정답 및 해설은 제외
            if "정답" in pdf_file.name or "해설" in pdf_file.name:
                continue
            
            # 단원명 매핑
            unit = unit_mapping.get(pdf_file.name, pdf_file.stem)
            
            pdf_files.append({
                'filepath': str(pdf_file),
                'subject': '수학',
                'unit': unit,
                'publisher': publisher,
                'filename': pdf_file.name
            })
    
    return pdf_files


def process_single_pdf(
    pipeline: RAGPipeline,
    pdf_info: Dict[str, str],
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    단일 PDF 파일 처리
    
    Args:
        pipeline: RAG 파이프라인
        pdf_info: PDF 파일 정보
        dry_run: True면 실제 처리 없이 정보만 출력
    
    Returns:
        처리 결과 딕셔너리
    """
    logger.info(f"Processing: {pdf_info['filename']}")
    logger.info(f"  Publisher: {pdf_info['publisher']}")
    logger.info(f"  Unit: {pdf_info['unit']}")
    
    if dry_run:
        return {
            'status': 'skipped',
            'reason': 'dry_run',
            **pdf_info
        }
    
    try:
        # PDF OCR 처리 및 벡터 DB 저장
        result = pipeline.process_textbook(
            file_path=pdf_info['filepath'],
            subject=pdf_info['subject'],
            unit=pdf_info['unit']
        )
        
        logger.info(f"✅ Success: {result['processed_chunks']} chunks processed")
        logger.info(f"   Tokens: {result['total_tokens']:,}")
        logger.info(f"   Cost: ${result.get('estimated_cost', result.get('estimated_cost_usd', 0)):.6f}")
        
        return {
            'status': 'success',
            **pdf_info,
            **result
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing {pdf_info['filename']}: {str(e)}")
        return {
            'status': 'error',
            'error': str(e),
            **pdf_info
        }


def main():
    """메인 처리 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='3학년 교과서 PDF를 OCR 처리하여 벡터 DB에 저장'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='실제 처리 없이 처리할 파일 목록만 출력'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='처리할 파일 개수 제한 (테스트용)'
    )
    parser.add_argument(
        '--unit',
        type=str,
        help='특정 단원만 처리 (예: "이차함수")'
    )
    parser.add_argument(
        '--publisher',
        type=str,
        help='특정 출판사만 처리 (예: "미래엔")'
    )
    
    args = parser.parse_args()
    
    # 교과서 디렉토리 경로 (ai-services/data/3rd_grade_textbook)
    textbook_dir = ai_services_dir / 'data' / '3rd_grade_textbook'
    
    if not textbook_dir.exists():
        logger.error(f"교과서 디렉토리를 찾을 수 없습니다: {textbook_dir}")
        sys.exit(1)
    
    # PDF 파일 목록 생성
    logger.info("PDF 파일 검색 중...")
    pdf_files = find_pdf_files(textbook_dir)
    
    # 필터링
    if args.unit:
        pdf_files = [f for f in pdf_files if args.unit in f['unit']]
    
    if args.publisher:
        pdf_files = [f for f in pdf_files if args.publisher in f['publisher']]
    
    if args.limit:
        pdf_files = pdf_files[:args.limit]
    
    logger.info(f"총 {len(pdf_files)}개 파일 발견")
    
    # Dry run - 파일 목록만 출력
    if args.dry_run:
        logger.info("\n=== 처리 대상 파일 목록 ===")
        for i, pdf_info in enumerate(pdf_files, 1):
            print(f"\n{i}. {pdf_info['filename']}")
            print(f"   출판사: {pdf_info['publisher']}")
            print(f"   단원: {pdf_info['unit']}")
            print(f"   경로: {pdf_info['filepath']}")
        
        print(f"\n총 {len(pdf_files)}개 파일")
        print("\n실제 처리하려면 --dry-run 없이 실행하세요.")
        return
    
    # RAG 파이프라인 초기화
    logger.info("\nRAG 파이프라인 초기화 중...")
    try:
        pipeline = RAGPipeline()
        logger.info("✅ 파이프라인 초기화 완료")
    except Exception as e:
        logger.error(f"❌ 파이프라인 초기화 실패: {e}")
        sys.exit(1)
    
    # 파일 처리
    logger.info(f"\n{'='*60}")
    logger.info(f"총 {len(pdf_files)}개 파일 처리 시작")
    logger.info(f"{'='*60}\n")
    
    results = []
    total_chunks = 0
    total_tokens = 0
    total_cost = 0.0
    
    for i, pdf_info in enumerate(pdf_files, 1):
        logger.info(f"\n[{i}/{len(pdf_files)}] {pdf_info['filename']}")
        
        result = process_single_pdf(pipeline, pdf_info, dry_run=False)
        results.append(result)
        
        if result['status'] == 'success':
            total_chunks += result['processed_chunks']
            total_tokens += result['total_tokens']
            total_cost += result.get('estimated_cost', result.get('estimated_cost_usd', 0))
        
        # API 속도 제한 방지 (1초 대기)
        if i < len(pdf_files):
            time.sleep(1)
    
    # 결과 요약
    logger.info(f"\n{'='*60}")
    logger.info("처리 완료!")
    logger.info(f"{'='*60}")
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    
    logger.info(f"\n📊 처리 결과:")
    logger.info(f"   성공: {success_count}개")
    logger.info(f"   실패: {error_count}개")
    logger.info(f"   총 청크: {total_chunks:,}개")
    logger.info(f"   총 토큰: {total_tokens:,}개")
    logger.info(f"   총 비용: ${total_cost:.6f}")
    
    # 실패 목록
    if error_count > 0:
        logger.info(f"\n❌ 실패한 파일:")
        for result in results:
            if result['status'] == 'error':
                logger.info(f"   - {result['filename']}: {result['error']}")
    
    # 결과를 JSON 파일로 저장
    output_file = ai_services_dir / 'data' / 'processing_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n💾 상세 결과가 {output_file}에 저장되었습니다.")
    
    # 벡터 DB 상태 확인
    logger.info("\n📊 벡터 DB 상태:")
    vector_stats = pipeline.vector_store.get_collection_info()
    logger.info(f"   총 문서: {vector_stats.get('total_documents', 0):,}개")
    logger.info(f"   컬렉션: {vector_stats.get('collection_name', 'N/A')}")
    if vector_stats.get('subjects'):
        logger.info(f"   과목: {', '.join(vector_stats['subjects'])}")
    if vector_stats.get('units'):
        logger.info(f"   단원: {', '.join(vector_stats['units'])}")


if __name__ == '__main__':
    main()
