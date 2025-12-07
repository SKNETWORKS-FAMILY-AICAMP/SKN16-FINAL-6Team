# AI 기반 코딩 테스트 플랫폼

**코딩 학습을 위한 AI 힌트 시스템 + RAG 챗봇 통합 플랫폼**

## 프로젝트 개요

이 프로젝트는 코딩 테스트 학습을 위한 통합 플랫폼으로, 세 가지 핵심 모듈로 구성됩니다:

1. **CHATBOT_MODEL** - RAG 기반 기술 문서 Q&A 챗봇
2. **HINT_MODEL** - LangGraph 기반 AI 힌트 생성 서비스
3. **SERVER** - Django 백엔드 + React 프론트엔드 통합 서버

---

## 폴더 구조

```
FINAL/
├── CHATBOT_MODEL/          # RAG 챗봇 모델
│   ├── crawler/            # Git, Python, Docker, AWS 문서 크롤러
│   ├── data/               # 크롤링된 원시 데이터
│   ├── experiments/        # RAG 파이프라인 실험
│   │   └── rag_pipeline/
│   │       └── langgraph_rag/  # LangGraph 기반 Adaptive RAG
│   ├── docs/               # 기술 문서
│   └── results/            # 평가 결과
│
├── HINT_MODEL/             # 힌트 생성 서비스 (Runpod Serverless)
│   ├── handler.py          # Runpod 핸들러
│   ├── hint_core.py        # 힌트 생성 핵심 로직 (LangGraph)
│   ├── code_analyzer_lite.py  # 경량 코드 분석기
│   ├── Dockerfile          # 컨테이너 이미지
│   └── server.py           # 로컬 테스트 서버
│
└── SERVER/                 # 웹 서버 (Django + React)
    ├── backend/            # Django 백엔드
    │   ├── apps/
    │   │   ├── coding_test/    # 코딩 테스트 핵심 앱
    │   │   ├── chatbot/        # RAG 챗봇 앱
    │   │   ├── authentication/ # 사용자 인증
    │   │   └── mypage/         # 마이페이지
    │   └── config/         # Django 설정
    ├── frontend/           # React 프론트엔드
    │   └── src/
    │       └── pages/      # 페이지 컴포넌트
    ├── scripts/            # 배포 스크립트
    ├── nginx/              # Nginx 설정
    ├── docs/               # 서버 문서
    └── docker-compose.yml  # Docker 구성
```

---

## 핵심 기능

### 1. CHATBOT_MODEL - RAG 챗봇

**기술 문서 Q&A 시스템**으로 Git, Python, Docker, AWS 문서에 대한 질의응답을 제공합니다.

**주요 특징:**
- **Hybrid Search**: Dense + Sparse + RRF Fusion
- **2-Stage Reranking**: BGE-reranker-v2-m3 + BGE-reranker-large
- **LangGraph Adaptive RAG**: Query Routing, Document Grading, Self-RAG
- **웹 검색 Fallback**: Tavily 연동
- **LangSmith 추적**: 워크플로우 디버깅

**성능:**
| 지표 | Optimized RAG | LangGraph RAG |
|------|---------------|---------------|
| Context Precision | 0.85 | 0.92 |
| Answer Relevancy | 0.90 | 0.95 |
| Hallucination Rate | 10% | 3% |
| 응답 속도 | 5초 | 7-10초 |

### 2. HINT_MODEL - AI 힌트 서비스

**LangGraph 기반 단계별 힌트 생성 시스템**으로 Runpod Serverless에서 실행됩니다.

**LangGraph 플로우:**
```
input → solution_match → purpose → parallel_analysis → branch
    → [조건부] → coh → prompt → llm_hint (+ 자기검증) → format → output
```

**주요 특징:**
- **정적 분석 6개 지표**: 구문 분석, 코드 복잡도, 변수 사용 등
- **LLM 분석 6개 지표**: 알고리즘 이해도, 코드 품질 등
- **난이도별 프리셋**: 초급/중급/고급
- **자기검증 시스템**: 힌트 품질 자동 검증

### 3. SERVER - 웹 플랫폼

**Django + React 기반 풀스택 웹 애플리케이션**입니다.

**백엔드 (Django):**
- 코딩 테스트 API (문제 목록, 제출, 채점)
- 12-메트릭 분석 시스템
- RAG 챗봇 API
- 사용자 인증 (Kakao OAuth 지원)
- 뱃지/업적 시스템
- 학습 로드맵

**프론트엔드 (React):**
- 문제 목록 페이지 (별점 시스템 0-3⭐)
- 코딩 테스트 페이지 (Monaco Editor)
- AI 힌트 요청 UI
- RAG 챗봇 인터페이스
- 마이페이지 (학습 통계)
- 관리자 패널

**인프라:**
- Blue/Green 무중단 배포
- GitHub Actions CI/CD
- Docker Compose 구성
- Nginx 리버스 프록시

---

## 기술 스택

### AI/ML
- **LLM**: GPT-4.1, GPT-4o-mini
- **Embedding**: BAAI/bge-m3
- **Reranking**: BGE-reranker-v2-m3, BGE-reranker-large
- **Framework**: LangGraph, LangChain
- **Vector DB**: ChromaDB
- **Evaluation**: RAGAS
- **Monitoring**: LangSmith

### Backend
- **Framework**: Django 4.x, Django REST Framework
- **Database**: PostgreSQL
- **Task Queue**: Celery
- **서버리스**: Runpod Serverless

### Frontend
- **Framework**: React 18
- **State**: Redux Toolkit
- **Routing**: React Router
- **Editor**: Monaco Editor
- **Charts**: Recharts

### DevOps
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions
- **Web Server**: Nginx
- **Deployment**: Blue/Green Strategy

---

## 시작하기

### 환경 변수 설정

```bash
# SERVER/.env
OPENAI_API_KEY=your_openai_key
RUNPOD_ENDPOINT_ID=your_endpoint_id
RUNPOD_API_KEY=your_runpod_key
DATABASE_URL=postgres://...
SECRET_KEY=your_django_secret
```

### 로컬 개발 환경

```bash
# 1. 백엔드 실행
cd SERVER/backend
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver

# 2. 프론트엔드 실행
cd SERVER/frontend
npm install
npm run dev
```

### Docker 실행

```bash
cd SERVER
docker-compose up -d
```

---

## 상세 문서

각 모듈별 상세 문서는 해당 폴더 내 README.md를 참조하세요:

- [CHATBOT_MODEL/README.md](./CHATBOT_MODEL/README.md) - RAG 챗봇 상세 가이드
- [HINT_MODEL/README.md](./HINT_MODEL/README.md) - 힌트 서비스 배포 가이드
- [SERVER/docs/](./SERVER/docs/) - 서버 문서 인덱스

---

## 팀 정보

**SK Networks AI Camp 16기 - 6팀**

---

## 라이선스

MIT License
