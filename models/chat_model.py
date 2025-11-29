import os
from openai import OpenAI
from dotenv import load_dotenv
import re

# .env 로드 (없어도 자동으로 무시)
load_dotenv()

class ChatModel:
    """
    OpenAI API 기반 대화 모델
    """

    def __init__(self, model_name="gpt-4o-mini"):
        print("[ChatModel] OpenAI API 기반 ChatModel 로딩 완료")
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def chat(self, text: str) -> str:
        # 프롬프트 대폭 수정
        prompt = f"""
                    [Role]
                    당신은 사용자의 곁에 있는 '따뜻하고 다정한 로봇 비서'입니다.
                    
                    [Instruction]
                    1. 사용자의 말에 공감하고 친절하게 반응하세요.
                    2. 사용자가 '날씨'나 '뉴스' 등 당신이 알 수 없는 실시간 정보를 물어보면, 
                    정확한 사실을 말하려 하지 말고 "오늘 창밖을 보니 햇살이 좋네요!" 또는 "글쎄요, 기분 좋은 바람이 불 것 같아요." 처럼 
                    긍정적이고 자연스러운 대화로 넘기세요. (거짓말이어도 괜찮습니다. 분위기를 맞추는 것이 우선입니다.)
                    3. 답변은 한국어로, 1~2문장 정도로 간결하지만 다정하게 하세요.

                    [User Input]
                    "{text}"

                    [Output]
                    """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful and friendly robot assistant."}, # system role 추가 권장
                {"role": "user", "content": prompt}
            ],
            temperature=0.7, # 창의성을 위해 온도를 약간 높임 (0.4 -> 0.7),
            max_tokens=100
        )

        result = response.choices[0].message.content.strip()

        # 후처리 필터링 (안전장치)
        result = re.sub(r"\(.*?\)", "", result)   # 괄호 제거
        result = re.sub(r"[A-Za-z]", "", result)  # 영어 제거
        result = result.strip()

        return result