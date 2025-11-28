import os
# [중요] Mac 충돌 방지를 위한 환경변수 설정 (반드시 import torch 전에)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# 기존 모듈 import 유지
from models.intent_classifier import IntentClassifier
from models.translator import Translator
from models.normalizer import Normalizer
from models.chat_model import ChatModel
from models.rag.personal_rag import PersonalRAG
from models.rag.personal_response import PersonalResponse
from models.rag.behavior_detector import BehaviorDetector
from models.rag.decision_model import DecisionModel

class ModelContainer:
    _instance = None

    def __init__(self):
        print("📥 AI 모델 로딩 시작... (최초 1회)")
        
        # [중요] 맥북(MPS) 가속 지원 추가
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"  # 맥북 M1/M2/M3 전용 가속
        else:
            self.device = "cpu"
            
        print(f"🚀 실행 장치: {self.device}")

        self.intent_classifier = IntentClassifier()
        self.chat_model = ChatModel()
        self.translator = Translator()
        self.normalizer = Normalizer()

        self.rag = PersonalRAG()
        self.personal_response = PersonalResponse()
        self.behavior_detector = BehaviorDetector()
        self.decision_model = DecisionModel()

        print("👂 Whisper(STT) 모델 로딩 중...")

        BASE_MODEL_ID = "openai/whisper-small"
        ADAPTER_PATH = "./models/whisper-finetuned-v1"

        # [수정됨] Processor와 Model 변수 분리
        self.processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID, language="Korean", task="transcribe")
        
        # device_map을 사용하여 자동으로 장치 할당
        base_hf_model = WhisperForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID, 
            device_map=self.device
        )

        if os.path.exists(ADAPTER_PATH):
            self.stt_model = PeftModel.from_pretrained(base_hf_model, ADAPTER_PATH)
            print("✅ LoRA 어댑터 적용 완료!")
        else:
            print(f"⚠️ 어댑터 없음. 기본 모델 사용: {ADAPTER_PATH}")
            self.stt_model = base_hf_model

        # 모델을 평가 모드로 전환 (메모리 절약)
        self.stt_model.eval()

        self.waiting_for_decision = False
        self.pending_task = None
        print("✅ AI 모델 로딩 완료!")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance