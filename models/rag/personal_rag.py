import os
import re
import numpy as np
import faiss
from typing import List, Tuple
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class PersonalRAG:
    """
    개인 정보 텍스트 파일을 기반으로
    의미 검색을 수행하는 간단한 RAG 모듈.

    - 텍스트 파일을 여러 chunk로 나눔
    - 각 chunk를 OpenAI Embedding으로 벡터화
    - FAISS index에 저장
    - 질의(query)에 가장 관련 있는 chunk들을 반환
    """

    def __init__(
        self,
        profile_path: str = "./data/test_profile.txt",
        embedding_model: str = "text-embedding-3-large",
        top_k: int = 3,
    ):
        self.profile_path = profile_path
        self.embedding_model = embedding_model
        self.top_k = top_k

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 가 .env 에 설정되어 있지 않습니다.")
        self.client = OpenAI(api_key=api_key)

        # 내부 상태
        self.chunks: List[str] = []
        self.index = None  # FAISS index

        print("[PersonalRAG] 프로필 로드 및 인덱스 생성 중...")
        self._build_index()
        print(f"[PersonalRAG] 완료. chunk 개수: {len(self.chunks)}")

    # ---------------------------
    # 내부 유틸 함수들
    # ---------------------------
    def _load_profile_text(self) -> str:
        if not os.path.exists(self.profile_path):
            raise FileNotFoundError(f"profile 파일을 찾을 수 없습니다: {self.profile_path}")

        with open(self.profile_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text

    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Semantic chunking (개선안)
            - 문장을 절대로 제거하지 않는다
            - 문장 단위로 split
            - 문장 2~3개씩 하나의 chunk로 묶기
        """
        # 1. 문장 나누기
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

        chunks = []
        current = []
        MAX_SENT = 3  # 문장당 2~3개씩 묶기

        for s in sentences:
            current.append(s)
            if len(current) >= MAX_SENT:
                chunks.append(" ".join(current))
                current = []

        # 마지막 덩어리 추가
        if current:
            chunks.append(" ".join(current))

        return chunks

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        OpenAI embedding API 를 사용해 여러 문장을 임베딩.
        반환 shape: (N, D)
        """
        if not texts:
            return np.zeros((0, 0), dtype="float32")

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        vectors = [item.embedding for item in response.data]
        arr = np.array(vectors, dtype="float32")
        return arr

    def _build_index(self):
        # 1) 텍스트 로드
        text = self._load_profile_text()

        # 2) chunk로 나누기
        self.chunks = self._split_into_chunks(text)

        # 3) 임베딩
        embeddings = self._embed_texts(self.chunks)
        if embeddings.shape[0] == 0:
            raise RuntimeError("profile에서 유효한 chunk를 찾지 못했습니다.")

        dim = embeddings.shape[1]

        # 4) FAISS index 생성 (L2 distance)
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        self.index = index

    def extract_keywords(self, sentences: List[str]) -> List[str]:
        """
        문장 리스트에서 명사성 단어만 자동 추출.
        규칙 기반 간단 버전 (불용어 제거 포함).
        """
        text = " ".join(sentences)

        # 단순 명사 후보 추출 (한글 or 영어 단어)
        candidates = re.findall(r"[가-힣A-Za-z]{2,}", text)

        # 불용어 필터
        stopwords = ["사용자", "있다", "한다", "하고", "하며", "있는", "그래서"]
        filtered = [w for w in candidates if w not in stopwords]

        # 중복 제거
        return list(set(filtered))

    def auto_expand_query(self, query: str, num_similar=3) -> str:
        """
        embedding 기반 문장 유사도 검색 후
        주요 단어를 자동 추출하여 query 확장.
        """
        # 1) query embedding
        q_emb = self._embed_texts([query])
        if q_emb.shape[0] == 0:
            return query

        # 2) 프로필 문장에서 semantic similarity top-N 검색
        distances, indices = self.index.search(q_emb, num_similar)
        similar_sentences = [self.chunks[idx] for idx in indices[0] if idx < len(self.chunks)]

        # 3) 키워드 자동 추출
        keywords = self.extract_keywords(similar_sentences)

        # 4) query 확장 구성
        expanded_query = query + " " + " ".join(keywords)
        #print(f"[AutoExpandQuery] {expanded_query}")

        return expanded_query

    # ---------------------------
    # 외부에서 쓰는 메소드들
    # ---------------------------
    def search(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        if top_k is None:
            top_k = self.top_k
        # 수정: 쿼리가 너무 짧으면(3어절 미만) 확장을 하지 않고 원문 그대로 검색
        if len(query.split()) < 3:
            expanded_query = query
        else:
            # Step 4 자동 Query Expansion
            expanded_query = self.auto_expand_query(query)

        # 1) embedding
        q_emb = self._embed_texts([expanded_query])
        if q_emb.shape[0] == 0:
            return []

        # 2) 검색
        distances, indices = self.index.search(q_emb, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx], float(dist)))

        return results

    def build_context(self, query: str, top_k: int = None) -> str:
        """
        검색된 chunk들을 하나로 이어붙여,
        LLM에 넣을 'context' 문자열로 만들어줌.
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return ""

        # distance 기준으로 정렬 (가까운 것 우선)
        results = sorted(results, key=lambda x: x[1])

        context_parts = [chunk for chunk, _ in results]
        context_text = "\n\n".join(context_parts)
        return context_text
