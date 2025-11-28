from dataclasses import dataclass
from typing import Literal


IntentType = Literal["robot_command", "dialog"]


@dataclass
class IntentResult:
    intent: IntentType
    reason: str


class IntentClassifier:
    def __init__(self):
        self.robot_keywords = [
            "가져와", "가져다줘", "집어줘", "잡아줘", "옮겨줘", "전달해줘", "줘", "가져다", "비상"
        ]

    def classify(self, text: str) -> IntentResult:
        text = text.strip()

        if any(kw in text for kw in self.robot_keywords):
            return IntentResult("robot_command", "로봇 명령 동사 발견")
        
        return IntentResult("dialog", "기본 대화")
