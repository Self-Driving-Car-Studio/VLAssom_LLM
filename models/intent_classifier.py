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
            "비상"
        ]

    def classify(self, text: str) -> IntentResult:
        text = text.strip()

        if any(kw in text for kw in self.robot_keywords):
            return IntentResult("robot_command", "로봇 명령 동사 발견")
        
        return IntentResult("dialog", "기본 대화")