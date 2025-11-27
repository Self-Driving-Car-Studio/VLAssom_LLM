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

        # [추가] Action Map의 Key들을 로드 (AI가 이 중에서 고르게 하기 위함)
        self.valid_keys = []
        try:
            with open("data/action_map.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                self.valid_keys = list(data.keys()) 
        except FileNotFoundError:
            print("[PersonalResponse] Warning: action_map.json not found.")

    def generate(self, user_input: str, context: str) -> str:
        # Key 목록을 프롬프트에 주입
        keys_str = ", ".join(self.valid_keys)

        prompt = f"""
                        당신은 로봇 비서입니다. 
                        사용자 프로필과 입력을 바탕으로 행동을 제안하세요.

                        [사용자 프로필]
                        {context}

                        [사용자 입력]
                        {user_input}

                        [가능한 행동 Key 목록]
                        {keys_str}

                        [작성 규칙]
                        1. "사용자 입력"에 구체적인 물건이나 행동 요청(예: 연필, 타이레놀)이 있다면, 프로필보다 입력을 **최우선**으로 따르세요.
                        2. 입력이 모호할 때만 프로필을 참고해서 제안하세요.
                        3. 제안 멘트 뒤에 " || Key"를 붙이세요. (Key는 위 목록에 있는 것만 사용)
                        4. 적절한 Key가 없으면 " || NONE"을 붙이세요.

                        [예시]
                        - 입력: "나 머리 아파" (프로필: 두통엔 타이레놀) -> "타이레놀 가져다 드릴까요? || serve_tylenol"
                        - 입력: "연필 필요해" (프로필: 비타민 선호) -> "연필 가져다 드릴까요? || bring pencil" (입력 우선!)
                """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7 
        )

        return response.choices[0].message.content.strip()