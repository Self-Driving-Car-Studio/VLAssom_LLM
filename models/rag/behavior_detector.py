import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class BehaviorDetector:
    """
    사용자의 대화 입력이 '로봇 행동'을 요구하는지 판단하는 모듈.
    (다국어 지원: lang='ko' or 'en')
    """

    # ------------------------------------------------------------------
    # 1. 언어별 키워드 정의 (Class Constants)
    # ------------------------------------------------------------------
    ACTION_KEYWORDS_KO = [
        # 1. 기본 요청/이동 동사
        "가져", "집어", "줘", "도와줘", "찾아줘", "내놔", "필요해", "필요",
        "손이 안 닿아", "열어줘", "닫아줘", "올려줘", "내려줘", "움직여", "밀어줘",
        # 2. 생리적 욕구
        "배고파", "목말라", "출출해", "입이 심심해",
        # 3. 건강/통증
        "아파", "두통", "머리", "열나", "몸살", "컨디션", "약", "진통제", "타이레놀",
        # 4. 활력/건강관리
        "피곤", "힘들", "지쳐", "기운", "활력", "비타민", "영양제",
        # 5. 작업/학습
        "공부", "메모", "적을", "쓰게", "필기", "기록", "숙제", "연필", "볼펜"
    ]

    ACTION_KEYWORDS_EN = [
        # 1. Basic Requests / Action Verbs
        "bring", "fetch", "get", "give", "help", "find", "need", "hand me",
        "cannot reach", "open", "close", "lift", "lower", "move", "push", "assist",
        # 2. Physiological Needs
        "hungry", "thirsty", "starving", "peckish", "famished", "snack", "drink", "eat",
        # 3. Health / Pain
        "pain", "headache", "hurt", "sick", "fever", "ache", "stomach", 
        "condition", "medicine", "pill", "tylenol",
        # 4. Vitality / Fatigue
        "tired", "exhausted", "weary", "fatigue", "drained", "energy", "weak", "vitamin",
        # 5. Work / Study
        "study", "note", "write", "homework", "pencil", "pen"
    ]

    def __init__(self, model_name="gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing.")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    # -----------------------
    # 1차: keyword rule 기반
    # -----------------------
    def rule_based(self, text: str, lang: str) -> bool:
        t = text.lower().strip()
        
        # 언어에 따른 키워드 선택
        keywords = self.ACTION_KEYWORDS_EN if lang == "en" else self.ACTION_KEYWORDS_KO

        for kw in keywords:
            if kw in t:
                return True
        return False

    # -----------------------
    # 2차: LLM 기반 확정 판단
    # -----------------------
    def llm_based(self, text: str, lang: str) -> bool:
        # 언어별 프롬프트 분기
        if lang == "en":
            prompt = f"""
            You are a robot control decision module.
            Determine if the following sentence requires **checking the robot's manual (profile)** for an action or advice.
            
            [Criteria - YES]
            1. Physical Assistance Request: "Give me water", "Open the door".
            2. Physical Symptoms/Status: "Stomach ache", "I'm so tired", "I'm hungry".
            3. Inquiry for Solution: "What should I do?", "What should I eat?"

            [Criteria - NO]
            1. Simple Greetings/Small Talk: "Hello", "I'm bored", "Nice weather".
            2. Exclamations: "Wow", "Really?"

            Sentence: "{text}"
            Answer only with YES or NO.
            """
        else:
            prompt = f"""
            당신은 로봇 판단 제어 모듈입니다.
            아래 문장이 **로봇의 매뉴얼(프로필)을 확인해야 하는 상황**인지 판단하세요.
            
            [판단 기준 - YES]
            1. 물리적 도움 요청: "물 줘", "문 열어", "약 가져와"
            2. 신체적 증상/상태 호소: "배 아파", "너무 피곤해", "배고파"
            3. 해결책 문의: "이럴 땐 어떻게 해?", "나 뭐 먹어야 할까?"

            [판단 기준 - NO]
            1. 단순 인사/잡담: "안녕", "심심해", "오늘 날씨 좋네"
            2. 감탄사: "와 웃기다", "정말?"

            문장: "{text}"
            정답을 YES 또는 NO 로만 말하세요.
            """

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0
            )
            answer = res.choices[0].message.content.strip().upper()
            return "YES" in answer
            
        except Exception as e:
            print(f"[BehaviorDetector] Error: {e}")
            # 에러 발생 시 보수적으로 True(행동 필요)로 판단하거나 False로 처리
            return False

    # -----------------------
    # 최종 판단 함수
    # -----------------------
    def detect(self, text: str, lang: str = "ko") -> bool:
        """
        Args:
            text (str): 사용자 입력
            lang (str): 'ko' 또는 'en'
        Returns:
            bool: 행동/프로필 확인이 필요하면 True
        """
        # 1) Rule 간단 체크
        if self.rule_based(text, lang):
            return True

        # 2) Rule 미스 → LLM에게 최종 판단
        return self.llm_based(text, lang)