#!/usr/bin/env python3
"""
Gemini API 직접 테스트
"""
import os
from dotenv import load_dotenv

# .env 로드
load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')
print(f'API Key: {api_key[:20]}...')

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

print('\n✅ Gemini 클라이언트 초기화...')
client = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.7,
    max_output_tokens=100
)

print('✅ 간단한 메시지 테스트...')
messages = [HumanMessage(content="안녕하세요! 간단하게 인사해주세요.")]

try:
    response = client.invoke(messages)
    print(f'\n✅ 응답 받음:')
    print(f'{response.content}')
except Exception as e:
    print(f'\n❌ 오류: {e}')
    import traceback
    traceback.print_exc()
