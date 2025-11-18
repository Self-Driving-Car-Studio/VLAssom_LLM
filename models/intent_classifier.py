from dataclasses import dataclass
from typing import Literal


IntentType = Literal["robot_command", "medicine_info", "dialog"]


@dataclass
class IntentResult:
    intent: IntentType
    reason: str


class IntentClassifier:
    def __init__(self):
        self.robot_keywords = [
            "가져와", "가져다줘", "집어줘", "잡아줘", "옮겨줘", "전달해줘", "줘"
        ]

        self.medicine_keywords = [
            "약", "복용", "부작용", "성분", "용량", "먹는 법", "먹는법"
        ]

        self.question_keywords = ["뭐야", "알려줘", "어떻게", "왜", "언제"]

    def classify(self, text: str) -> IntentResult:
        text = text.strip()

        if any(kw in text for kw in self.medicine_keywords):
            return IntentResult("medicine_info", "약 관련 단어 발견")

        if any(kw in text for kw in self.robot_keywords):
            return IntentResult("robot_command", "로봇 명령 동사 발견")

        if any(kw in text for kw in self.question_keywords):
            return IntentResult("dialog", "질문 패턴")

        return IntentResult("dialog", "일반 문장")
