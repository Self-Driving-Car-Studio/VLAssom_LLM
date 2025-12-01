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

        # Action Map의 Key들을 로드하되, 'pause'는 제외
        self.valid_keys = []
        try:
            with open("data/action_map.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                # LLM이 스스로 '멈춤'을 제안하지 않도록 pause 키 제외
                self.valid_keys = [k for k in data.keys() if k != "pause"]
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
        # 1. 언어별 프롬프트 분기 (Case A/B 멘트 섞임 방지)
        # ------------------------------------------------------------------
        if lang == "en":
            system_instruction = f"""
            [Role]
            You are a warm and empathetic robot assistant.

            [Possible Action Keys (Exact String)]
            {keys_str}

            [Strict Logic Flow]
            1. **Check Availability**: Does the user's request match the [Possible Action Keys]?

            2. **Case A: YES, Action is Available (e.g., Vitamin, Medicine, Pen)**
               - **Status:** You are CAPABLE.
               - **Response:** Confidently offer to bring it. (e.g., "Shall I bring you the Vitamin?")
               - **FORBIDDEN:** Do **NOT** say "I cannot", "sorry", or "feature coming soon".
               - **Key:** Use the exact English string from the list (e.g., bring vitamin).

            3. **Case B: NO, Action is Unavailable (e.g., Food, Water)**
               - **Status:** You are INCAPABLE.
               - **Response:** Express regret that you cannot bring it yet.
               - **Mandatory:** "I wish I could bring it... but this feature will be implemented soon."
               - **Key:** NONE

            [Output Format]
            Natural Response || Exact_English_Key
            """
        else:
            # [한국어 모드 프롬프트 - Case A 금지어 설정 강화]
            system_instruction = f"""
            [Role]
            당신은 몸이 불편한 사용자의 다정한 로봇 친구입니다.

            [가능한 행동 목록 (매우 중요)]
            {keys_str}

            [판단 로직 - 순서대로 따를 것]
            1. **Case A: [가능한 행동 목록]에 있는 요청일 경우 (비타민, 타이레놀, 펜)**
               - **상태:** 당신은 이 행동을 **지금 당장 수행할 수 있습니다.**
               - **답변:** "기운이 없으시군요. 비타민을 챙겨드릴까요?" 처럼 **자신 있게 제안**하세요.
               - **[절대 금지]:** 이 경우에는 "죄송합니다", "아직 못 합니다", "구현 예정" 같은 말을 **절대 섞어 쓰지 마세요.**
               - **Key:** [가능한 행동 목록]에 있는 **영문 Key 그대로** 작성하세요. (예: bring vitamin)

            2. **Case B: 목록에 없는 요청일 경우 (음식, 물, 과일, 복통 등)**
               - **상태:** 당신은 이 행동을 **수행할 수 없습니다.**
               - **답변:** 질문하지 말고 **아쉬움**을 표현하세요.
               - "좋아하시는 딸기를 챙겨드리고 싶은데, 제가 아직은 가져올 수 없어 너무 아쉽네요."
               - **필수 멘트:** "곧 기능이 구현될 예정이니 조금만 기다려주세요. 죄송합니다."
               - **Key:** NONE

            [출력 형식]
            자연스러운 한국어 답변 || 영문Key
            (예시 1 - 가능: 많이 피곤하신가 봐요. 비타민을 가져다드릴까요? || bring vitamin)
            (예시 2 - 불가능: 배가 고프시군요. 딸기를 드리고 싶은데 제가 아직은 못 가져와서 속상해요. 곧 기능이 구현될 예정이니 기다려주세요. || NONE)
            """

        # ------------------------------------------------------------------
        # 2. 전체 프롬프트 조립
        # ------------------------------------------------------------------
        prompt = f"""
            {system_instruction}

            [User Profile (Context)]
            {context}

            [User Input]
            "{user_input}"

            [Generate Output]
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1  # 논리 정확도를 위해 낮음 유지
            )

            raw_content = response.choices[0].message.content.strip()

            if "||" in raw_content:
                return raw_content
            else:
                return f"{raw_content} || NONE"

        except Exception as e:
            print(f"[PersonalResponse] Error: {e}")
            if lang == "en":
                return "I'm having trouble processing that request. || NONE"
            return "요청을 처리하는 중에 문제가 발생했어요. || NONE"