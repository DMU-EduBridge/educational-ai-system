#!/usr/bin/env python3
"""
통합 교과서 PDF를 단원별로 자동 분류하여 벡터 DB에 저장하는 스크립트
"""

import sys
import os
from pathlib import Path
import json
from typing import List, Dict, Any, Optional
import re
from dotenv import load_dotenv

# 프로젝트 루트 경로 설정
script_dir = Path(__file__).resolve().parent
ai_services_dir = script_dir.parent
project_root = ai_services_dir.parent

# .env 파일 로드
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded .env from {env_path}")

sys.path.insert(0, str(ai_services_dir))

from src.main import RAGPipeline
from src.rag.document_processor import DocumentProcessor, Document
from src.utils.logger import get_logger

logger = get_logger(__name__)


# 중학교 3학년 수학 단원 키워드 매핑
UNIT_KEYWORDS = {
    "실수와 그 계산": [
        "제곱근", "실수", "무리수", "유리수", "순환소수", "근호", "실수의 대소관계",
        "제곱근의 성질", "실수와 그 연산"
    ],
    "이차방정식": [
        "이차방정식", "인수분해", "근의 공식", "완전제곱식", "판별식", 
        "이차방정식의 풀이", "이차방정식의 활용"
    ],
    "이차함수": [
        "이차함수", "포물선", "꼭짓점", "축", "y절편", "x절편",
        "이차함수의 그래프", "이차함수의 최댓값", "이차함수의 최솟값",
        "이차함수의 활용"
    ],
    "삼각비": [
        "삼각비", "사인", "코사인", "탄젠트", "sin", "cos", "tan",
        "특수각", "삼각비의 활용", "삼각비의 값"
    ],
    "원의 성질": [
        "원의 성질", "현", "접선", "중심각", "원주각", "내접원", "외접원",
        "원주각의 성질", "접선의 성질"
    ],
    "통계": [
        "산포도", "분산", "표준편차", "상관관계", "상관표", "산점도",
        "대푯값", "평균", "중앙값", "최빈값"
    ]
}


def detect_unit(text: str) -> Optional[str]:
    """
    텍스트 내용을 분석하여 해당하는 단원을 감지
    
    Args:
        text: 분석할 텍스트
    
    Returns:
        감지된 단원명 또는 None
    """
    # 각 단원별 키워드 매칭 점수 계산
    scores = {}
    
    for unit, keywords in UNIT_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            # 키워드가 텍스트에 포함된 횟수 계산
            count = len(re.findall(keyword, text, re.IGNORECASE))
            score += count
        
        scores[unit] = score
    
    # 가장 높은 점수를 가진 단원 선택
    if max(scores.values()) > 0:
        detected_unit = max(scores, key=scores.get)
        return detected_unit
    
    return None


def process_with_unit_detection(
    pipeline: RAGPipeline,
    pdf_path: str,
    subject: str = "수학"
) -> Dict[str, Any]:
    """
    PDF를 처리하면서 각 청크의 단원을 자동 감지
    
    Args:
        pipeline: RAG 파이프라인
        pdf_path: PDF 파일 경로
        subject: 과목명
    
    Returns:
        처리 결과
    """
    logger.info(f"Processing {pdf_path} with unit detection...")
    
    try:
        # DocumentProcessor를 사용하여 PDF 로드
        doc_processor = DocumentProcessor()
        
        # PDF를 전체 텍스트로 로드 (임시로 "통합교과서" 사용)
        documents = doc_processor.load_textbook(
            file_path=pdf_path,
            subject=subject,
            unit="통합교과서"  # 임시값
        )
        
        logger.info(f"Loaded {len(documents)} initial documents")
        
        # 각 문서를 청크로 분할하고 단원 감지
        all_chunks = []
        for doc in documents:
            # chunk_text는 List[str]을 반환
            chunk_texts = doc_processor.chunk_text(doc.content, chunk_size=1000, overlap=100)
            
            for i, chunk_text in enumerate(chunk_texts):
                # 청크 내용으로 단원 감지
                detected_unit = detect_unit(chunk_text)
                
                # 메타데이터 생성
                chunk_metadata = doc.metadata.copy()
                chunk_metadata['chunk_index'] = i
                chunk_metadata['total_chunks'] = len(chunk_texts)
                chunk_metadata['chunk_size'] = len(chunk_text)
                
                # 단원이 감지된 경우 메타데이터 업데이트
                if detected_unit:
                    chunk_metadata['unit'] = detected_unit
                    chunk_metadata['auto_detected'] = True
                else:
                    # 감지 실패 시 기본값 유지
                    chunk_metadata['unit'] = "통합교과서"
                    chunk_metadata['auto_detected'] = False
                
                # Document 객체 생성
                chunk_doc = Document(
                    content=chunk_text,
                    metadata=chunk_metadata
                )
                all_chunks.append(chunk_doc)
        
        logger.info(f"Created {len(all_chunks)} chunks with unit detection")
        
        # 단원별 통계
        unit_stats = {}
        for chunk in all_chunks:
            unit = chunk.metadata.get('unit', '미분류')
            unit_stats[unit] = unit_stats.get(unit, 0) + 1
        
        logger.info("Unit distribution:")
        for unit, count in sorted(unit_stats.items()):
            logger.info(f"  - {unit}: {count} chunks")
        
        # 임베딩 생성 및 벡터 DB 저장
        embeddings = pipeline.embeddings_manager.generate_embeddings(
            [chunk.content for chunk in all_chunks]
        )
        
        success = pipeline.vector_store.add_documents(all_chunks, embeddings)
        
        if success:
            logger.info(f"✅ Successfully processed {len(all_chunks)} chunks")
            return {
                'status': 'success',
                'total_chunks': len(all_chunks),
                'unit_distribution': unit_stats,
                'file': pdf_path
            }
        else:
            raise Exception("Failed to add documents to vector store")
            
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        return {
            'status': 'error',
            'error': str(e),
            'file': pdf_path
        }


def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("통합 교과서 단원 자동 분류 스크립트")
    print("=" * 80)
    
    # RAG 파이프라인 초기화
    logger.info("Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    
    # 처리할 PDF 파일 경로
    textbook_dir = ai_services_dir / "data" / "3rd_grade_textbook"
    pdf_file = textbook_dir / "비상 중3 수학 교과서.pdf"
    
    if not pdf_file.exists():
        logger.error(f"PDF file not found: {pdf_file}")
        return
    
    # 기존 벡터 DB 백업 안내
    vector_db_path = Path(os.getenv('CHROMA_DB_PATH', str(ai_services_dir / "data" / "vector_db")))
    print(f"\n⚠️  경고: 기존 벡터 DB ({vector_db_path})를 덮어씁니다.")
    response = input("계속하시겠습니까? (yes/no): ")
    
    if response.lower() != 'yes':
        print("작업이 취소되었습니다.")
        return
    
    # 기존 컬렉션 삭제
    try:
        pipeline.vector_store.delete_collection()
        logger.info("Deleted existing collection")
    except Exception as e:
        logger.warning(f"Could not delete collection: {e}")
    
    # 새 컬렉션 생성
    pipeline.vector_store = pipeline.vector_store.__class__(
        persist_directory=str(vector_db_path),
        collection_name="textbook_embeddings"
    )
    
    # PDF 처리
    result = process_with_unit_detection(
        pipeline=pipeline,
        pdf_path=str(pdf_file),
        subject="수학"
    )
    
    # 결과 출력
    print("\n" + "=" * 80)
    print("처리 결과")
    print("=" * 80)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 검증: 각 단원별로 검색 테스트
    print("\n" + "=" * 80)
    print("단원별 검색 테스트")
    print("=" * 80)
    
    for unit in UNIT_KEYWORDS.keys():
        try:
            docs = pipeline.retriever.retrieve_documents(
                query=f"{unit} 개념",
                subject="수학",
                unit=unit,
                k=1
            )
            
            if docs:
                print(f"✅ {unit}: {len(docs)} 문서 검색 성공")
            else:
                print(f"❌ {unit}: 검색 결과 없음")
        except Exception as e:
            print(f"❌ {unit}: 오류 - {str(e)}")
    
    print("\n✅ 모든 작업이 완료되었습니다!")


if __name__ == "__main__":
    main()
