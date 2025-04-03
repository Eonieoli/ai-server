# 베이스 이미지 선택
# GPU 사용 시: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
# CPU 사용 시: python:3.12-slim
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 환경 변수 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 환경 설정
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# requirements.txt 파일 먼저 복사
COPY requirements.txt .

# 애플리케이션 종속성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 필요한 디렉토리 생성
RUN mkdir -p /app/models/downloads /app/temp /app/logs

# 실행 권한 설정 및 줄바꿈 문자 변환
RUN chmod +x /app/entrypoint.sh && \
    sed -i 's/\r$//' /app/entrypoint.sh

# 포트 노출
EXPOSE 8000

# 진입점 설정
ENTRYPOINT ["/app/entrypoint.sh"]

# 기본 명령
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
