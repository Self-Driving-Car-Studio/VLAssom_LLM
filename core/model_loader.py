from models.intent_classifier import IntentClassifier
from models.translator import Translator
from models.normalizer import Normalizer
from models.chat_model import ChatModel

# 신규 RAG + 행동 판단 모듈들
from models.rag.personal_rag import PersonalRAG
from models.rag.personal_response import PersonalResponse
from models.rag.behavior_detector import BehaviorDetector
from models.rag.decision_model import DecisionModel

class ModelContainer:
    _instance = None

    def __init__(self):
        print("📥 AI 모델 로딩 시작... (최초 1회)")
        self.intent_classifier = IntentClassifier()
        self.chat_model = ChatModel()
        self.translator = Translator()
        self.normalizer = Normalizer()

        # 신규 AI 모듈들
        self.rag = PersonalRAG()
        self.personal_response = PersonalResponse()
        self.behavior_detector = BehaviorDetector()
        self.decision_model = DecisionModel()

        # 제안 후 응답 상태
        self.waiting_for_decision = False
        self.pending_task = None   # normalized single_task 저장용
        print("✅ AI 모델 로딩 완료!")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance