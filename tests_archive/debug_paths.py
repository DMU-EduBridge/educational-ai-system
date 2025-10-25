"""경로 디버깅 스크립트"""
import os
import sys
from pathlib import Path

# backend에서 실행한다고 가정
os.chdir("/Users/hyunjong_kim/Desktop/KHJ/dongyang/2025_2nd/graduate_project/educational-ai-system/backend")

# ai-services를 path에 추가 (backend/main.py와 동일)
project_root = Path(__file__).resolve().parent.parent
ai_services_path = project_root / 'ai-services'
sys.path.insert(0, str(ai_services_path))

print(f"현재 작업 디렉토리: {os.getcwd()}")
print(f"Project root: {project_root}")
print(f"AI services path: {ai_services_path}")

from src.utils.config import get_settings

settings = get_settings()
print(f"\n.env 파일 경로: {settings.Config.env_file}")
print(f"CHROMA_DB_PATH 설정값: {settings.chroma_db_path}")
print(f"절대 경로: {Path(settings.chroma_db_path).resolve()}")
print(f"경로 존재: {Path(settings.chroma_db_path).exists()}")

# 올바른 경로
correct_path = "/Users/hyunjong_kim/Desktop/KHJ/dongyang/2025_2nd/graduate_project/educational-ai-system/ai-services/data/vector_db"
print(f"\n올바른 경로: {correct_path}")
print(f"올바른 경로 존재: {Path(correct_path).exists()}")
