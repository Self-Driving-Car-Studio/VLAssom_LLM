import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class DecisionModel:
    """
    사용자의 짧은 응답을 YES / NO / UNKNOWN 으로 분류하는 모듈.
    Rule 기반 + LLM 기반 하이브리드 방식.
    """

    YES_KEYWORDS = [
        "네"
    ]

    NO_KEYWORDS = [
        "아니오"
    ]

    def __init__(self, model_name="gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing.")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    # --------------------------
    # 1. Rule 기반 판단
    # --------------------------
    def rule_based(self, text: str) -> str:
        t = text.strip().lower()

        for kw in self.YES_KEYWORDS:
            if kw in t:
                return "YES"

        for kw in self.NO_KEYWORDS:
            if kw in t:
                return "NO"

        return "UNKNOWN"

    # --------------------------
    # 2. 모호할 때 LLM 판단
    # --------------------------
    def llm_based(self, text: str) -> str:
        prompt = f"""
                    다음 문장이 제안된 행동을 수락하는지 YES/NO/UNKNOWN 으로 판단하세요.

                    문장: "{text}"

                    - YES: 제안/행동을 수락함
                    - NO: 제안/행동을 거절함
                    - UNKNOWN: 모호함

                    정답만 YES 또는 NO 또는 UNKNOWN으로 출력하세요.
                    """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4,
            temperature=0.0
        )

        answer = response.choices[0].message.content.strip().upper()

        if answer.startswith("YES"):
            return "YES"
        if answer.startswith("NO"):
            return "NO"
        return "UNKNOWN"

    # --------------------------
    # 최종 결정 함수
    # --------------------------
    def decide(self, text: str) -> str:
        # 1) Rule 기반 먼저
        rule = self.rule_based(text)
        if rule != "UNKNOWN":
            return rule

        # 2) 모호하면 LLM 기반 판단
        return self.llm_based(text)
