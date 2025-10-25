#!/usr/bin/env python3
"""
벡터 DB의 교과서 데이터를 사용하여 문제 생성 테스트
"""
import sys
sys.path.insert(0, 'ai-services')

from src.main import RAGPipeline

def main():
    print('✅ RAG Pipeline 초기화')
    pipeline = RAGPipeline()

    print('\n📊 벡터 DB 상태:')
    stats = pipeline.vector_store.get_collection_info()
    print(f'   총 문서: {stats.get("total_documents", 0)}개')

    print('\n📝 문제 생성 중...')
    questions = pipeline.generate_questions(
        subject='수학',
        unit='이차방정식',
        difficulty='medium',
        count=1
    )

    print(f'\n✅ 문제 생성 완료!\n')
    q = questions[0]
    print(f'【문제】')
    print(f'{q["question"]}\n')
    print(f'【선택지】')
    for i, opt in enumerate(q['options'], 1):
        marker = '✓' if i == q['correct_answer'] else ' '
        print(f'{marker} {i}. {opt}')
    print(f'\n【정답】 {q["correct_answer"]}번')
    print(f'\n【해설】')
    print(q['explanation'])

if __name__ == '__main__':
    main()
