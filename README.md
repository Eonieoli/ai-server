# AI 이미지 분석 서버

사용자가 업로드한 이미지를 분석하여 품질과 심미적 특성을 자동으로 평가하는 FastAPI 기반 AI 서버입니다.

## 기능

- 이미지 URL을 받아 AI 모델로 분석
- 다음 6가지 기준으로 이미지 평가:
  - 구도(Composition)
  - 선명도(Sharpness)
  - 노이즈(Noise)
  - 노출(Exposure)
  - 색감(Color Harmony)
  - 심미성(Aesthetics)
- 각 기준별 점수(1-10)와 설명 제공
- CPU 및 GPU 환경 모두 지원

## 기술 스택

- **웹 프레임워크**: FastAPI
- **AI 모델**: LLaVA-NeXT-Video-7B (Hugging Face)
- **컨테이너화**: Docker
- **이미지 처리**: PIL, Torch Vision

## 설치 및 실행 방법

### 로컬 환경에서 실행

1. 저장소 클론:
   ```bash
   git clone <repository-url>
   cd ai-server
   ```

2. 가상 환경 설정:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. 환경 변수 설정:
   `.env` 파일을 수정하여 필요한 환경 변수 설정

4. 애플리케이션 실행:
   ```bash
   uvicorn app.main:app --reload
   ```

### Docker로 실행

#### CPU 환경

```bash
docker-compose up ai-server-cpu
```

#### GPU 환경

```bash
docker-compose up ai-server-gpu
```

## API 사용법

### 이미지 분석 API

**엔드포인트**: `POST /api/v1/image/analyze`

**요청 본문**:
```json
{
  "image_url": "https://example.com/image.jpg"
}
```

**응답**:
```json
{
  "composition": {
    "score": 8,
    "comment": "구도에 대한 평가 코멘트"
  },
  "sharpness": {
    "score": 7,
    "comment": "선명도에 대한 평가 코멘트"
  },
  "noise": {
    "score": 9,
    "comment": "노이즈에 대한 평가 코멘트"
  },
  "exposure": {
    "score": 6,
    "comment": "노출에 대한 평가 코멘트"
  },
  "color_harmony": {
    "score": 8,
    "comment": "색감에 대한 평가 코멘트"
  },
  "aesthetics": {
    "score": 7,
    "comment": "심미성에 대한 평가 코멘트"
  },
  "overall": {
    "score": 8,
    "comment": "이미지에 대한 종합적인 평가 코멘트"
  }
}
```

### 시스템 상태 확인 API

**엔드포인트**: `GET /api/v1/system/health`

**응답**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "gpu_available": true
}
```

## 테스트

```bash
pytest
```

## 환경 변수

- `USE_GPU`: GPU 사용 여부 (true/false)
- `GPU_DEVICE`: 사용할 GPU 장치 번호 (0부터 시작)
- `MODEL_NAME`: 사용할 Hugging Face 모델 이름
- `DEBUG`: 디버그 모드 활성화 여부

## 디렉토리 구조

```
ai-server/
│
├── app/                     # 애플리케이션 코드
│   ├── main.py              # FastAPI 애플리케이션 메인
│   ├── api/                 # API 관련 코드
│   ├── core/                # 핵심 설정 및 유틸리티
│   ├── models/              # 모델 정의
│   ├── services/            # 비즈니스 로직
│   ├── schemas/             # Pydantic 스키마
│   └── utils/               # 유틸리티 함수
│
├── models/                  # 모델 파일 저장
│   └── downloads/           # 다운로드된 모델 저장
│
├── data/                    # 데이터 저장
│
├── tests/                   # 테스트 코드
│
├── temp/                    # 임시 파일 저장
│
├── .env                     # 환경 변수
├── docker-compose.yml       # Docker 구성
├── Dockerfile               # CPU용 Dockerfile
├── Dockerfile.gpu           # GPU용 Dockerfile
└── requirements.txt         # 의존성 목록
```

## 라이센스

[라이센스 정보 추가]
