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
        # 1. 기본 요청/이동 동사
        "가져", "집어", "줘", "도와줘", "찾아줘", "내놔", "필요해", "필요",
        "손이 안 닿아", "열어줘", "닫아줘", "올려줘", "내려줘", "움직여", "밀어줘",
        "필요한데",

        # 2. 생리적 욕구 (기존)
        "배고파", "목말라", "출출해", "입이 심심해",
        "아파", "두통", "머리", "열나", "몸살", "컨디션",

        # 3. 건강/통증 (Target: bring the tylenol)
        "아파", "두통", "머리", "열나", "몸살", "컨디션", 
        "약", "진통제", "타이레놀",

        # 4. 활력/건강관리 (Target: bring vitamin)
        "피곤", "힘들", "지쳐", "기운", "활력",
        "비타민", "영양제", "건강",

        # 5. 작업/학습 (Target: bring pencil)
        "공부", "메모", "적을", "쓰게", "필기", "기록", "숙제",
        "연필", "볼펜", "펜", "샤프"
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
    # 2. [수정] LLM 프롬프트 강화: 증상 호소도 행동 필요로 간주
    def llm_based(self, text: str) -> bool:
        prompt = f"""
                    아래 문장이 로봇의 물리적인 도움이나 제안이 필요한 상황인지 판단하세요.

                    [판단 기준]
                    1. 명시적 요청: "물 줘", "문 열어" → YES
                    2. 상태 호소: "목 말라(물 필요)", "머리 아파(약 필요)", "너무 어두워(불 켜기 필요)" → YES
                    3. 단순 감정/잡담: "심심해", "오늘 날씨 좋네", "기분 좋아" → NO

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
        return "YES" in answer

    # -----------------------
    # 최종 판단 함수
    # -----------------------
    def detect(self, text: str) -> bool:
        # 1) Rule 간단 체크
        if self.rule_based(text):
            return True

        # 2) Rule 미스 → LLM에게 최종 판단
        return self.llm_based(text)
