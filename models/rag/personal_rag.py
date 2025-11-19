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
        embedding_model: str = "text-embedding-3-small",
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
        아주 단순하게:
        - 빈 줄(두 줄 이상 공백) 기준으로 문단 단위 split
        - 너무 짧은 건 버림
        """
        raw_chunks = re.split(r"\n\s*\n", text)
        chunks = [c.strip() for c in raw_chunks if len(c.strip()) > 5]
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

    # ---------------------------
    # 외부에서 쓰는 메소드들
    # ---------------------------
    def search(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        query 에 대해 가장 비슷한 chunk들을 반환.
        return: [(chunk_text, distance), ...]
        """
        if top_k is None:
            top_k = self.top_k

        # query embedding
        q_emb = self._embed_texts([query])  # (1, D)
        if q_emb.shape[0] == 0:
            return []

        # FAISS 검색
        distances, indices = self.index.search(q_emb, top_k)
        distances = distances[0]
        indices = indices[0]

        results: List[Tuple[str, float]] = []
        for idx, dist in zip(indices, distances):
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
