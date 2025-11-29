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

    def generate(self, user_input: str, context: str) -> str:
        """
        Returns:
            (response_text, action_key) 튜플을 반환합니다.
            - response_text: 사용자에게 보여줄 답변 (예: "타이레놀 드릴까요?")
            - action_key: 로봇 제어 코드 (예: "bring the tylenol" 또는 "NONE")
        """
        # Key 목록 문자열화
        keys_str = ", ".join(self.valid_keys)

        # 프롬프트 대폭 수정: 제약 조건 강화 및 우선순위 명시
        prompt = f"""
                        당신은 거동이 불편한 사용자를 돕는 로봇 비서입니다.
                        제공된 [사용자 프로필]을 **엄격히 준수**하여 답변과 행동 Key를 결정하세요.

                        [사용자 프로필 (가장 중요)]
                        {context}

                        [사용자 입력]
                        "{user_input}"

                        [가능한 행동 Key 목록]
                        {keys_str}

                        [필수 판단 로직]
                        1. 1. **상황 일치 확인 (중요)**: 프로필에 있는 '제약 사항'이 현재 사용자가 말한 '상황'과 일치하는지 먼저 확인하세요.
                            - 사용자가 "배고파"라고 했으나 프로필은 "배 아플 때"에 대한 내용이라면, 이를 적용하지 말고 배고픔에 대해 응답하세요.
                            - 단, 프로필에 "배고픔"에 대한 직접적인 지시가 없다면 일반적인 공감 대화를 하세요.
                        2. **매칭 확인**: 사용자의 요청이 위 [가능한 행동 Key 목록]의 행동과 정확히 일치할 때만 Key를 반환하세요.
                        3. **부정 제약 준수**: 프로필에서 "하지 말라"거나 "효과 없다"고 한 행동은 절대 Key로 반환하지 마세요.
                           - 예: 배 아플 때(복통)는 타이레놀 금지 → 위로의 말과 함께 Key는 NONE 출력.
                        4. **대안 제시**: 행동 Key가 없더라도 프로필에 해결책(예: 따뜻한 물, 휴식)이 있다면 말로 제안하세요.

                        [출력 형식]
                        자연스러운 한국어 답변 || 선택된 Key

                        [Few-shot 예시]
                        Q: "나 머리 아파." (프로필: 두통엔 타이레놀)
                        A: 네, 두통에는 타이레놀을 드릴 수 있습니다. 가져다 드릴까요? || bring the tylenol

                        Q: "배가 너무 아파." (프로필: 복통엔 약 없음, 따뜻한 물 권장 / 타이레놀 금지)
                        A: 배가 많이 아프시군요. 복통에는 약을 드리기 어렵습니다. 대신 따뜻한 물을 좀 드시는 건 어떨까요? || NONE

                        Q: "피곤해." (프로필: 피곤할 땐 비타민)
                        A: 많이 피곤해 보이세요. 비타민을 좀 챙겨 드릴까요? || bring vitamin

                        Q: "어떻게 해야 할까?" (맥락: 머리 아픔)
                        A: 두통이 있으시면 타이레놀을 가져다 드릴 수 있습니다. || bring the tylenol
                        
                        이제 답변하세요.
                """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        raw_content = response.choices[0].message.content.strip()

        # 내부 검증 및 포맷팅 (안전장치)
        # LLM이 실수를 하더라도, router.py가 터지지 않게 "||" 구조를 보장해줍니다.
        if "||" in raw_content:
            return raw_content # 형식이 맞으면 그대로 반환
        else:
            # 구분자가 없으면 Key를 NONE으로 간주하고 포맷을 맞춰줌
            return f"{raw_content} || NONE"