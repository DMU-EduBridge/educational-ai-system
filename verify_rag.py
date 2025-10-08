
import sys
import os
import json

# ai-services/src 경로를 sys.path에 추가하여 모듈을 임포트할 수 있도록 함
sys.path.insert(0, os.path.abspath('ai-services/src'))

from rag.vector_store import VectorStore
from utils.logger import setup_logger

def verify_rag_processing():
    """
    VectorStore를 초기화하고 컬렉션 정보를 출력하여
    문서가 처리되고 추가되었는지 확인합니다.
    """
    setup_logger('verify_rag')
    try:
        print("VectorStore를 초기화하여 컬렉션 정보를 확인합니다...")
        # 기본 persist_directory는 './data/vector_db'이지만,
        # 프로젝트 루트에서 실행하므로 'ai-services/data/vector_db'로 지정해야 합니다.
        vector_store = VectorStore(persist_directory="ai-services/data/vector_db")
        
        print("컬렉션 정보를 가져오는 중...")
        collection_info = vector_store.get_collection_info()
        
        print("\n--- RAG 처리 확인 결과 ---")
        # JSON 형식으로 예쁘게 출력
        print(json.dumps(collection_info, indent=2, ensure_ascii=False))
        print("--------------------------\n")

        total_docs = collection_info.get('total_documents', 0)
        if total_docs > 0:
            print(f"✅ 성공: 벡터 스토어에서 {total_docs}개의 문서를 찾았습니다.")
            print("RAG 처리가 성공적으로 완료된 것으로 보입니다.")
        else:
            print("❌ 실패: 벡터 스토어에서 문서를 찾을 수 없습니다.")
            print("RAG 처리에 실패했을 수 있습니다.")

    except Exception as e:
        print(f"확인 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    verify_rag_processing()
