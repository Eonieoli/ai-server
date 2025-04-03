import requests
import json

# AI 서버 URL
ai_server_url = "http://70.12.130.141:8000/api/v1/image/analyze"

# 이미지 URL (S3 등에 업로드된 이미지 URL)
image_url = "https://picscore-s3.s3.ap-northeast-2.amazonaws.com/permanent/932d650d-16ec-49df-ae88-fa09c4db99f0.jpg"

# 요청 데이터
data = {
    "image_url": image_url
}

# API 호출
response = requests.post(ai_server_url, json=data)

# 응답 처리
if response.status_code == 200:
    result = response.json()
    print("분석 결과:", json.dumps(result, indent=2, ensure_ascii=False))
else:
    print(f"오류 발생: {response.status_code} - {response.text}")