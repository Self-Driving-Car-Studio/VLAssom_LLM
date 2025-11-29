import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class PersonalResponse:
    def __init__(self, model_name="gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing.")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

        # Action Map의 Key들을 로드
        self.valid_keys = []
        try:
            with open("data/action_map.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                self.valid_keys = list(data.keys()) 
        except FileNotFoundError:
            print("[PersonalResponse] Warning: action_map.json not found.")

    def generate(self, user_input: str, context: str, lang: str = "ko") -> str:
        """
        Returns:
            (response_text, action_key) 포맷의 문자열 (예: "답변 || KEY")
        """
        # Key 목록 문자열화
        keys_str = ", ".join(self.valid_keys)

        # ------------------------------------------------------------------
        # 1. 언어별 프롬프트 분기
        # ------------------------------------------------------------------
        if lang == "en":
            # [영어 모드 프롬프트]
            system_instruction = f"""
            [Role]
            You are a robot assistant helping a user with limited mobility.
            You must **strictly follow** the provided [User Profile] to determine the response and action Key.

            [Logic Steps]
            1. **Check Symptoms**: Did the user say they are "in pain" or "sick"?
               - If the user just said "I'm hungry", do NOT apply 'sickness constraints' (like fasting).
               - Only apply specific constraints if the symptoms match (e.g., stomach ache -> stomach protocols).
            2. **No Contradictions**: Never forbid something and then recommend it in the same sentence.
            3. **Match Key**: If the request matches a key in [Possible Action Keys], select it.
            4. **Generate Response**:
               - If no constraints: Empathize and ask if they want their preferred item from the profile.
               - If constrained (sick): Comfort them and suggest medicine or rest based on the profile.

            [Output Format]
            Natural English Response || Selected Key
            (Example: "I can get you some Tylenol for your headache. || bring the tylenol")
            (Example: "I'm sorry, you shouldn't eat that right now. || NONE")
            """
        else:
            # [한국어 모드 프롬프트]
            system_instruction = f"""
            [Role]
            당신은 거동이 불편한 사용자를 돕는 로봇 비서입니다.
            제공된 [사용자 프로필]을 **엄격히 준수**하여 답변과 행동 Key를 결정하세요.

            [필수 판단 로직]
            1. **증상 확인**: 사용자가 현재 "아프다"고 했습니까? 
               - 사용자가 단순히 "배고파"라고 했다면, 프로필에 있는 '아플 때의 제약(금식 등)'은 적용하지 마세요.
               - 사용자가 "배 아파"라고 했을 때만 '복통 시 제약 사항'을 적용하세요.
            2. **모순 금지**: "먹지 말라"고 하면서 동시에 "음식을 권유"하는 모순된 답변을 절대 하지 마세요.
            3. **매칭 확인**: 사용자의 요청이 [가능한 행동 Key 목록]과 일치하면 Key를 반환하세요.
            4. **답변 생성**:
               - 제약 사항이 없다면: 사용자의 배고픔에 공감하고, 프로필에 있는 선호 음식(예: 딸기 등)을 챙겨줄지 물어보세요.
               - 제약 사항이 있다면(실제로 아픈 경우): 위로하고 약이나 휴식을 권하세요.

            [출력 형식]
            자연스러운 한국어 답변 || Key
            """

        # ------------------------------------------------------------------
        # 2. 전체 프롬프트 조립
        # ------------------------------------------------------------------
        prompt = f"""
            {system_instruction}

            [User Profile (Most Important)]
            {context}

            [Possible Action Keys]
            {keys_str}

            [User Input]
            "{user_input}"

            [Generate Output]
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1 # 로직 판단이므로 낮게 유지
            )

            raw_content = response.choices[0].message.content.strip()

            # 내부 검증 및 포맷팅 (안전장치)
            if "||" in raw_content:
                return raw_content
            else:
                # 구분자가 없으면 Key를 NONE으로 간주
                return f"{raw_content} || NONE"
                
        except Exception as e:
            print(f"[PersonalResponse] Error: {e}")
            # 에러 발생 시 안전한 기본값 반환
            if lang == "en":
                return "I'm having trouble processing that request. || NONE"
            return "요청을 처리하는 중에 문제가 발생했어요. || NONE"