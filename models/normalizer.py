import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class Normalizer:
    def __init__(self, model_name="gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        
        # Action Map의 'Key'들만 리스트로 로드
        self.valid_keys = []
        try:
            with open("data/action_map.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                self.valid_keys = list(data.keys()) # 예: ["serve_tylenol", "bring_water"]
        except FileNotFoundError:
            print("[Normalizer] Warning: action_map.json not found.")

    def normalize(self, text: str) -> str:
        # Key 목록을 프롬프트에 주입
        keys_str = ", ".join(self.valid_keys)

        prompt = f"""
        사용자의 요청을 분석하여, 아래 [가능한 Key 목록] 중 가장 적절한 하나를 선택하세요.
        
        [가능한 Key 목록]
        {keys_str}

        [사용자 요청 (영어 번역됨)]
        "{text}"

        [규칙]
        1. 절대 다른 말(설명, 문장)을 붙이지 마세요.
        2. 오직 목록에 있는 'Key' 단어 하나만 출력하세요.
        3. 목록에 적절한 것이 없다면 "UNKNOWN"이라고 출력하세요.
        
        정답:
        """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0  # 창의성 0% (정확한 매칭 위해)
        )

        # 공백 제거 후 반환
        return response.choices[0].message.content.strip()