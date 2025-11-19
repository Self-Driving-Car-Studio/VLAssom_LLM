import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class PersonalResponse:
    """
    PersonalRAG에서 추출된 context와 사용자 입력을 바탕으로
    개인 맞춤형 제안을 만들어내는 모듈.
    """
    def __init__(self, model_name="gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing.")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, user_input: str, context: str) -> str:
        """
        context: RAG가 찾아준 개인 정보
        user_input: 사용자의 입력 문장
        """

        prompt = f"""
                당신은 개인 맞춤형 로봇 비서입니다.
                아래는 사용자의 개인 정보입니다. 이를 참고하여 '자연스럽고 친절한 한 문장'으로 답변하세요.
                그리고 답변은 항상 어떤 행동을 제안하는 형태여야 합니다.

                [사용자 정보]
                {context}

                [사용자 입력]
                {user_input}

                [답변 가이드라인]
                - 행동을 제안하는 형태로 한 문장으로 작성
                - 영어, 괄호, 번역문 금지
                - 너무 많은 정보를 주지 말 것
                - 사용자가 직접 할 수 없는 행동을 대신 제안해도 됨(로봇 비서이므로)
                - 예시: "딸기를 가져다 드릴까요?", "따뜻한 차를 준비해드릴까요?"

                지금 답변하세요:
                """

        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.6
        )

        return res.choices[0].message.content.strip()
