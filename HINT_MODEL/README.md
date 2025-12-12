# Hint Service (Runpod Serverless)

Django와 분리된 독립적인 힌트 생성 서비스입니다.
Runpod Serverless에서 실행되어 힌트 생성 로직을 처리합니다.

## 구조

```
hint_service/
├── handler.py           # Runpod Serverless 핸들러
├── hint_core.py         # 힌트 생성 핵심 로직
├── code_analyzer_lite.py # 경량 코드 분석기
├── requirements.txt     # Python 의존성
├── Dockerfile          # 컨테이너 이미지 빌드
├── data/               # 문제 데이터 (선택사항)
└── README.md           # 이 파일
```

## 배포 방법

### 1. Docker 이미지 빌드

```bash
cd hint_service
docker build -t hint-service:latest .
```

### 2. Docker Hub 또는 Runpod Registry에 푸시

```bash
# Docker Hub
docker tag hint-service:latest your-username/hint-service:latest
docker push your-username/hint-service:latest

# 또는 Runpod 콘솔에서 직접 빌드
```

### 3. Runpod Serverless Endpoint 생성

1. [Runpod 콘솔](https://www.runpod.io/console/serverless) 접속
2. "New Endpoint" 클릭
3. 설정:
   - Container Image: `your-username/hint-service:latest`
   - GPU Type: CPU 또는 저렴한 GPU (힌트 생성은 CPU로 충분)
   - Min Workers: 0 (비용 절약) 또는 1 (Cold Start 방지)
   - Max Workers: 5 (트래픽에 따라 조정)
4. 환경 변수 설정:
   - `OPENAI_API_KEY`: OpenAI API 키
   - `OPENAI_MODEL`: 사용할 모델 (기본: gpt-4.1)

### 4. Django 환경 변수 설정

```bash
# .env 또는 환경 변수
RUNPOD_ENDPOINT_ID=your-endpoint-id
RUNPOD_API_KEY=your-runpod-api-key
```

## API

### 힌트 요청

```json
{
  "input": {
    "type": "hint",
    "problem_id": "1",
    "user_code": "print('hello')",
    "star_count": 0,
    "preset": "초급",
    "custom_components": {
      "libraries": true,
      "code_example": true,
      "step_by_step": true,
      "complexity_hint": true,
      "edge_cases": true,
      "improvements": true
    },
    "previous_hints": [],
    "problem_data": {
      "title": "두 수의 합",
      "description": "두 정수를 입력받아...",
      "solutions": [...]
    }
  }
}
```

### 핑 요청 (Keep-Alive)

```json
{
  "input": {
    "type": "ping"
  }
}
```

### 응답 형식

```json
{
  "success": true,
  "data": {
    "hint": "힌트 내용...",
    "hint_content": {...},
    "hint_branch": "B",
    "static_metrics": {...},
    "llm_metrics": {...}
  },
  "error": null
}
```

## Keep-Alive 설정

Cold Start 방지를 위해 5분마다 핑을 보내는 것을 권장합니다.

### Celery Beat 사용 (Django)

```python
# settings.py 또는 celery.py
CELERY_BEAT_SCHEDULE = {
    'runpod-keep-alive': {
        'task': 'apps.coding_test.tasks.keep_runpod_warm',
        'schedule': timedelta(minutes=5),
    },
}
```

### 수동 테스트

```python
from apps.coding_test.tasks import manual_keep_alive
manual_keep_alive()
```

## 비용

- **CPU만 사용 시**: ~$0.0001/초
- **GPU 사용 시**: ~$0.0002-0.0005/초 (GPU 타입에 따라)
- **Keep-Alive 핑**: 거의 무료 (짧은 실행 시간)

## 트러블슈팅

### Cold Start가 너무 길다

- Min Workers를 1로 설정
- 또는 Keep-Alive 주기를 3분으로 단축

### API 오류

- OPENAI_API_KEY 환경 변수 확인
- Runpod 콘솔에서 로그 확인

### 타임아웃

- Django hint_proxy.py의 timeout 값 조정 (기본 90초)
