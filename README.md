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
│   ├── langgraph_rag/      # LangGraph 13개 노드 구현
│   │   ├── config.py       # 설정 로더
│   │   ├── state.py        # State 정의 (23 fields)
│   │   ├── nodes.py        # 13개 노드 구현
│   │   ├── graph.py        # LangGraph 워크플로우
│   │   ├── tools.py        # 검색/Reranking 도구
│   │   └── main.py         # CLI 진입점
│   ├── config/             # RAG 설정 파일
│   ├── prompts/            # LLM 프롬프트 템플릿
│   ├── artifacts/          # 벡터 DB (Git LFS)
│   ├── serve_unified.py    # FastAPI 서버
│   └── README.md           # 상세 실행 가이드
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

**LangGraph 기반 기술 문서 Q&A 시스템**으로 Git, Python 문서에 대한 질의응답을 제공합니다.

**13개 노드 워크플로우:**
1. Intent Classifier → 2. Load User Context (개인화) → 3. Query Router
→ 4. Hybrid Retrieve → 5-6. Two-Stage Reranking → 7. Grade Documents
→ 8. Transform Query → 9. Web Search (fallback) → 10. Generate (GPT-4o)
→ 11. Hallucination Check → 12. Answer Grading → 13. Suggest Related Questions

**주요 특징:**
- **Hybrid Search**: Dense (bge-m3) + Sparse (BM25) + RRF Fusion
- **Two-Stage Reranking**: 속도(Stage1) + 정확도(Stage2) 균형
- **Self-RAG**: 문서 품질 평가, 환각 체크, 답변 평가
- **Constitutional AI**: Anthropic 원칙 기반 프롬프트
- **Personalization**: 사용자 학습 이력 기반 추천

**RAGAS 평가 성능:**
| 지표 | 점수 |
|------|------|
| Context Precision | 84.61% |
| Context Recall | 92.08% |
| Faithfulness | 86.81% |
| Answer Relevancy | 82.40% |
| Answer Correctness | 69.12% |

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
- **LLM**: GPT-4o, GPT-4o-mini
- **Embedding**: BAAI/bge-m3 (다국어 지원)
- **Reranking**: BGE-reranker-v2-m3 (빠름), BGE-reranker-large (정밀)
- **Framework**: LangGraph, LangChain
- **Vector DB**: ChromaDB (Git LFS)
- **Evaluation**: RAGAS
- **Search**: Tavily (웹 검색)

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

### 1. 환경 변수 설정

`.env` 파일은 보안상 GitHub에 포함되지 않습니다. `.env.example`을 참고하여 직접 생성하세요.

**SERVER 환경 변수:**
```bash
cd SERVER
cp .env.example .env
```

| 변수명 | 설명 |
|--------|------|
| `DJANGO_SECRET_KEY` | Django 시크릿 키 |
| `DB_PASSWORD` | MySQL 비밀번호 |
| `RUNPOD_AI_URL` | 힌트 AI 서버 URL (RunPod) |
| `RUNPOD_CHATBOT_URL` | RAG 챗봇 서버 URL (RunPod) |
| `VITE_API_BASE_URL` | 프론트엔드 API 주소 |

**CHATBOT_MODEL 환경 변수:**
```bash
cd CHATBOT_MODEL
cp .env.example .env
```

| 변수명 | 설명 |
|--------|------|
| `OPENAI_API_KEY` | OpenAI API 키 (GPT-4o) |
| `TAVILY_API_KEY` | Tavily API 키 (웹 검색, 선택) |

### 2. Git LFS 설정 (CHATBOT_MODEL 사용 시 필수)

CHATBOT_MODEL의 벡터 데이터베이스는 Git LFS로 관리됩니다.

```bash
# Git LFS 설치
git lfs install

# 벡터 DB 파일 다운로드
git lfs pull
```

### 3. DB 초기화

DB 백업 파일(`db_backup.sql`)은 GitHub에 포함되지 않습니다. 필요시 팀원에게 요청하세요.

```bash
# DB 컨테이너 실행 후 백업 파일 복원
cd SERVER
docker-compose up -d db
docker-compose exec -T db mysql -u hint_user -p hint_system < db_backup.sql
```

### 4. 로컬 개발 환경

```bash
# 백엔드 실행
cd SERVER/backend
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver

# 프론트엔드 실행 (별도 터미널)
cd SERVER/frontend
npm install
npm run dev
```

### 5. Docker 실행

```bash
cd SERVER
docker-compose up -d --build
docker-compose exec backend python manage.py migrate
docker-compose exec backend python manage.py collectstatic --noinput
```

### 6. CHATBOT_MODEL 서버 실행 (선택)

```bash
cd CHATBOT_MODEL
pip install -r requirements.txt
python serve_unified.py --rag-type langgraph --port 8080
```

서버가 `http://localhost:8080`에서 실행됩니다.

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
