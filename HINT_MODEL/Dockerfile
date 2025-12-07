# Runpod Hint Service Dockerfile
#
# 빌드: docker build -t hint-service .
# Runpod 배포: Runpod 콘솔에서 이 이미지를 사용하여 Serverless Endpoint 생성

FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY handler.py .
COPY hint_core.py .
COPY code_analyzer_lite.py .

# 데이터 디렉토리 생성 (필요시 문제 데이터 포함)
COPY data/ ./data/

# 환경 변수 (Runpod 콘솔에서 설정)
# OPENAI_API_KEY: OpenAI API 키
# OPENAI_MODEL: 사용할 모델 (기본: gpt-4.1)

# 포트 노출 (Runpod Serverless는 자동 처리)
EXPOSE 8000

# 엔트리포인트
CMD ["python", "-u", "handler.py"]
