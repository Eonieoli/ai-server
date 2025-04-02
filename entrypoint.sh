#!/bin/bash
set -e

# 필요한 디렉토리 생성
mkdir -p /app/models/downloads
mkdir -p /app/temp
mkdir -p /app/logs

# 디렉토리 권한 설정
chmod -R 755 /app/models/downloads
chmod -R 755 /app/temp
chmod -R 755 /app/logs

# 환경 변수 출력 (디버깅용)
echo "Environment: "
echo "USE_GPU: $USE_GPU"

# GPU 설정 메시지 출력
if [ "$USE_GPU" = "true" ]; then
  echo "Running with GPU enabled"
else
  echo "Running with CPU only"
fi

# 명령 실행
exec "$@"
