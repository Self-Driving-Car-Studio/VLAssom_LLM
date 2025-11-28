from models.intent_classifier import IntentClassifier
from models.translator import Translator
from models.normalizer import Normalizer
from models.chat_model import ChatModel

# 신규 RAG + 행동 판단 모듈들
from models.rag.personal_rag import PersonalRAG
from models.rag.personal_response import PersonalResponse
from models.rag.behavior_detector import BehaviorDetector
from models.rag.decision_model import DecisionModel
import whisper
import torch

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import os

class ModelContainer:
    _instance = None

    def __init__(self):
        print("📥 AI 모델 로딩 시작... (최초 1회)")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 실행 장치: {self.device}")

        self.intent_classifier = IntentClassifier()
        self.chat_model = ChatModel()
        self.translator = Translator()
        self.normalizer = Normalizer()

        # 신규 AI 모듈들
        self.rag = PersonalRAG()
        self.personal_response = PersonalResponse()
        self.behavior_detector = BehaviorDetector()
        self.decision_model = DecisionModel()

        print("👂 Whisper(STT) 모델 로딩 중...")
        self.stt_model = whisper.load_model("small")

        # 기본 모델 (Hugging Face 버전)
        BASE_MODEL_ID = "openai/whisper-small" # 학습때 쓴 베이스 모델과 같아야 함
        ADAPTER_PATH = "../models/whisper-finetuned-v1" # 경로 확인 필수!

        self.dys_processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID, language="Korean", task="transcribe")
        base_hf_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID, device_map=self.device)

        # LoRA 어댑터 결합
        if os.path.exists(ADAPTER_PATH):
            self.dys_model = PeftModel.from_pretrained(base_hf_model, ADAPTER_PATH)
            print("✅ LoRA 어댑터 적용 완료!")
        else:
            print(f"⚠️ 경고: 어댑터 경로({ADAPTER_PATH})가 없습니다. 기본 모델로 동작합니다.")
            self.dys_model = base_hf_model

        # 제안 후 응답 상태
        self.waiting_for_decision = False
        self.pending_task = None   # normalized single_task 저장용
        print("✅ AI 모델 로딩 완료!")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance