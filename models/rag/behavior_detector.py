import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class BehaviorDetector:
    """
    사용자의 대화 입력이 '로봇 행동'을 요구하는지 판단하는 모듈.
    rule-based → llm 두 단계로 처리
    """

    ACTION_KEYWORDS = [
        "가져", "집어", "줘", "배고파", "목말라",
        "손이 안 닿아", "도와줘", "열어줘", "닫아줘",
        "올려줘", "내려줘", "움직여", "밀어줘"
    ]

    def __init__(self, model_name="gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name

    # -----------------------
    # 1차: keyword rule 기반
    # -----------------------
    def rule_based(self, text: str) -> bool:
        t = text.lower().strip()

        for kw in self.ACTION_KEYWORDS:
            if kw in t:
                return True
        return False

    # -----------------------
    # 2차: LLM 기반 확정 판단
    # -----------------------
    def llm_based(self, text: str) -> bool:
        prompt = f"""
                    아래 문장이 로봇의 실제 행동이 필요한 상황인지 판단하세요.
                    예: 물 가져다줘, 도와줘, 손이 안 닿아, 목말라 → 행동 필요(YES)
                    예: 심심해, 기분이 안 좋아 → 행동 불필요(NO)

                    문장: "{text}"

                    정답을 YES 또는 NO 로만 말하세요.
                    """

        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0
        )

        answer = res.choices[0].message.content.strip().upper()
        return answer.startswith("YES")

    # -----------------------
    # 최종 판단 함수
    # -----------------------
    def detect(self, text: str) -> bool:
        # 1) Rule 간단 체크
        if self.rule_based(text):
            return True

        # 2) Rule 미스 → LLM에게 최종 판단
        return self.llm_based(text)
