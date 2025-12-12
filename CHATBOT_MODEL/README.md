# 개발 학습 도우미 챗봇 - RAG 서버

LangGraph 기반 RAG 챗봇 시스템

## 프로젝트 구조

```
CHATBOT_MODEL/
├── langgraph_rag/        # LangGraph 핵심 로직
│   ├── config.py         # 설정 로더
│   ├── state.py          # State 정의 (23 fields)
│   ├── nodes.py          # 13개 노드 구현
│   ├── graph.py          # LangGraph 워크플로우
│   ├── tools.py          # 검색/Reranking 도구
│   └── main.py           # CLI 진입점
├── config/               # RAG 설정 파일
│   └── enhanced.yaml     # 하이퍼파라미터 설정
├── prompts/              # LLM 프롬프트 템플릿
│   └── system_v2.txt     # Constitutional AI 프롬프트
├── artifacts/            # 벡터 데이터베이스
│   └── chroma_db/        # ChromaDB 인덱스 (Git LFS)
├── serve_unified.py      # FastAPI 서버
├── requirements.txt      # Python 의존성
└── .env.example          # 환경변수 템플릿
```

## 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정

`.env.example`을 `.env`로 복사하고 API 키 설정:

```bash
cp .env.example .env
```

`.env` 파일 내용:
```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here  # 웹 검색 (선택)
```

### 3. RAG 서버 실행

```bash
python serve_unified.py --rag-type langgraph --port 8080
```

서버가 `http://localhost:8080`에서 실행됩니다.

### 4. API 테스트

```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "git rebase란 무엇인가요?",
    "user_id": "test_user",
    "chat_history": []
  }'
```

## 시스템 아키텍처

### LangGraph RAG 워크플로우 (13개 노드)

1. **intent_classifier** - 질문 범위 판단 (Git/Python)
2. **load_user_context** - 사용자 컨텍스트 로드 (개인화)
3. **query_router** - 질문 라우팅 (vectorstore/websearch/direct)
4. **hybrid_retrieve** - Dense + Sparse 하이브리드 검색
5. **rerank_stage1** - 1차 Reranking (bge-reranker-v2-m3, 빠름)
6. **rerank_stage2** - 2차 Reranking (bge-reranker-large, 정밀)
7. **grade_documents** - 문서 관련성 평가
8. **transform_query** - 쿼리 재작성 (품질 미달 시)
9. **web_search** - Tavily 웹 검색 (폴백)
10. **generate** - LLM 답변 생성 (GPT-4o)
11. **hallucination_check** - 환각 검증
12. **answer_grading** - 답변 유용성 평가
13. **suggest_related_questions** - 관련 질문 + 상기 메시지 생성

### 핵심 특징

- **Hybrid Search**: Dense (벡터) + Sparse (BM25) 검색
- **Two-Stage Reranking**: 속도(Stage1) + 정확도(Stage2) 균형
- **Self-RAG**: 문서 품질 평가, 환각 체크, 답변 평가
- **Constitutional AI**: Anthropic의 원칙 기반 프롬프트
- **Personalization**: 사용자 선택 이력 기반 맞춤 추천

## 주요 설정 파라미터

`config/enhanced.yaml` 참조:

```yaml
chunking:
  chunk_size: 900         # Median 기반 선정
  chunk_overlap: 180      # 20% overlap

embedding:
  model_name: BAAI/bge-m3  # 다국어 임베딩

retrieval:
  hybrid_dense_top_k: 50   # Dense 검색 후보
  hybrid_sparse_top_k: 50  # Sparse 검색 후보
  rrf_k: 60                # RRF 상수
  rerank_top_k: 10         # 최종 문서 개수

llm:
  model_name: gpt-4o-mini  # 답변 생성 모델
  temperature: 0.2
```

## Git LFS 설정 (벡터 DB)

벡터 데이터베이스는 Git LFS로 관리됩니다:

```bash
git lfs install
git lfs pull
```

## 라이센스

MIT License
