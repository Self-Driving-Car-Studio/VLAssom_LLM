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
        prompt = f"""
                    다음 문장에 대해 한국어로 자연스럽게 한 문장만 답변하세요.
                    영어, 번역문, 괄호(), 역할극, 추가 질문을 절대 넣지 마세요.
                    사용자의 입력이 장난스럽거나 의미가 불명확하면
                    "필요한 것이 있으면 말씀해주세요." 라고 답하세요.

                    문장: {text}
                    답변:
                    """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=64
        )

        result = response.choices[0].message.content.strip()

        # 후처리 필터링 (안전장치)
        result = re.sub(r"\(.*?\)", "", result)   # 괄호 제거
        result = re.sub(r"[A-Za-z]", "", result)  # 영어 제거
        result = result.strip()

        return result