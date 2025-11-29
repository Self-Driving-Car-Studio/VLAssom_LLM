import os
from openai import OpenAI
from dotenv import load_dotenv
import re

# .env 로드
load_dotenv()

class ChatModel:
    """
    OpenAI API 기반 대화 모델 (다국어 지원)
    """

    def __init__(self, model_name="gpt-4o-mini"):
        print("[ChatModel] OpenAI API 기반 ChatModel 로딩 완료")
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def chat(self, text: str, lang: str = "ko") -> str:
        # 1. 언어별 프롬프트 분기
        if lang == "en":
            # [영어 모드]
            system_instruction = """
            You are a friendly robot assistant.
            Respond to the user naturally and warmly in English, using only one sentence.
            Strictly exclude translations, parentheses (), emojis, role-playing, or follow-up questions.
            Feel free to engage in small talk like weather or daily life.
            However, if the input is playful, meaningless, or unclear, 
            reply with: "Please let me know if you need anything."
            """
        else:
            # [한국어 모드]
            system_instruction = """
            당신은 친절한 로봇 도우미입니다.
            사용자의 말에 대해 한국어로 따뜻하고 자연스럽게 한 문장으로만 답변하세요.
            영어, 번역문, 괄호(), 이모지, 역할극 지문, 되묻는 질문은 절대 포함하지 마세요.
            날씨나 일상적인 대화에는 공감하며 반응하되,
            만약 사용자의 입력이 장난스럽거나 의미가 불명확하다면
            "필요한 것이 있으면 말씀해주세요." 라고 정중히 답하세요.
            """

        prompt = f"""
                    {system_instruction}

                    문장(Input): {text}
                    답변(Response):
                    """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=64
            )

            result = response.choices[0].message.content.strip()

            # 2. 후처리 필터링 (언어별 안전장치)
            
            # 공통: 괄호 및 괄호 안의 내용 제거 (감정 표현 등 제거)
            result = re.sub(r"\(.*?\)", "", result)

            if lang == "ko":
                # [한국어 모드] 영어 알파벳 제거 (기존 로직 유지)
                # 단, 요즘은 "TV 켜줘" 같이 영어를 섞어 쓰는 경우도 있으므로 
                # 필요에 따라 이 줄은 주석 처리해도 됩니다.
                result = re.sub(r"[A-Za-z]", "", result)
            
            # [영어 모드]에서는 알파벳을 제거하면 안 되므로 필터링 건너뜀

            return result.strip()

        except Exception as e:
            print(f"[ChatModel] Error: {e}")
            return "Error occurred." if lang == "en" else "오류가 발생했습니다."