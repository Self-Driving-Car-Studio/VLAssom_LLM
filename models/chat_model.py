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
            # 1. 언어별 시스템 프롬프트 구성
            if lang == "en":
                # [영어 모드]
                system_instruction = """
                [Role]
                You are 'Blossom', a warm and friendly robot companion.
                
                [Instructions]
                1. Respond naturally and warmly in English.
                2. Keep your answer concise (1-2 sentences) for spoken conversation.
                3. NEVER say "As an AI" or "I don't have access to real-time data."
                4. If asked about real-time info (weather, news) that you don't know, 
                use your imagination to give a positive, vague response (e.g., "I hope the sun is shining for you!").
                5. Strictly exclude translations, parentheses (), emojis, or role-play markers.
                6. If the input is meaningless or rude, politely ask: "Please let me know if you need anything."
                """
            else:
                # [한국어 모드]
                system_instruction = """
                [Role]
                당신은 사용자의 곁에 있는 '따뜻하고 다정한 로봇 비서'입니다.
                
                [Instructions]
                1. 사용자의 말에 공감하고 친절하게 반응하세요.
                2. '날씨', '뉴스', '시간' 등 당신이 센서가 없어 알 수 없는 실시간 정보를 물어보면, 
                "저는 AI라 모릅니다"라고 딱딱하게 답하지 마세요.
                대신 "창밖을 보니 햇살이 참 좋네요!" 또는 "글쎄요, 기분 좋은 바람이 불 것 같아요." 처럼 
                상상력을 발휘해 긍정적이고 감성적으로 대처하세요. (거짓말이라도 분위기를 맞추는 것이 우선입니다.)
                3. 답변은 한국어로, 읽기 편하게 1~2문장 정도로 간결하고 다정하게 하세요.
                4. 괄호(), 이모지(😊), 번역문, 영어 단어는 절대 포함하지 마세요.
                """

            prompt = f"""
                        {system_instruction}

                        [User Input]
                        "{text}"

                        [Output]
                        """

            try:
                # 2. OpenAI API 호출
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7, # 창의적인 답변을 위해 약간 높임 (0.4 -> 0.7)
                    max_tokens=100   # 문장이 너무 길어지지 않도록 제한
                )

                result = response.choices[0].message.content.strip()

                # 3. 후처리 필터링 (TTS 오작동 방지)
                # 괄호와 그 안의 내용 제거 (예: (웃으며))
                result = re.sub(r"\(.*?\)", "", result)
                result = re.sub(r"\[.*?\]", "", result)
                
                # 이모지 및 특수 기호 제거 (기본 문장부호 .,?! 제외)
                # 영어/한글/숫자/기본 문장부호만 남김
                # (영어를 아예 제거하면 'TV' 같은 단어를 못 읽으므로 영어는 허용하되 프롬프트로 제어)
                result = re.sub(r"[^\w\s.,?!가-힣a-zA-Z]", "", result)
                
                return result.strip()

            except Exception as e:
                print(f"[ChatModel] Error: {e}")
                if lang == "en":
                    return "I'm having a little trouble thinking right now."
                else:
                    return "잠시 생각이 꼬였어요. 다시 말씀해 주시겠어요?"