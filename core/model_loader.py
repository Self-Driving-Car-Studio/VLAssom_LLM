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
        print("📥 AI 모델 로딩 시작... (Medium + Small Dual Setup)")
        
        # 장치 설정
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"🚀 실행 장치: {self.device}")

        # 1. 기타 모델 로드
        self.intent_classifier = IntentClassifier()
        self.chat_model = ChatModel()
        self.translator = Translator()
        self.normalizer = Normalizer()

        self.rag = PersonalRAG()
        self.personal_response = PersonalResponse()
        self.behavior_detector = BehaviorDetector()
        self.decision_model = DecisionModel()

        # ---------------------------------------------------------
        # [핵심 수정] 1차(Medium) & 2차(Small+LoRA) 모델 분리 로딩
        # ---------------------------------------------------------
        print("👂 Whisper 모델 2종(Medium, Small) 로딩 중...")

        ID_MEDIUM = "openai/whisper-medium"
        ID_SMALL = "openai/whisper-small"
        ADAPTER_PATH = "./models/whisper-finetuned-v1"

        # Processor는 Medium 기준으로 하나만 생성 (Small과 토크나이저 호환됨)
        self.processor = WhisperProcessor.from_pretrained(ID_MEDIUM)

        # [1차] Medium 모델 (순정)
        print(f"   - Loading 1st Stage: {ID_MEDIUM}...")
        self.stt_model_medium = WhisperForConditionalGeneration.from_pretrained(
            ID_MEDIUM, 
            device_map=self.device
        )
        self.stt_model_medium.eval()

        # [2차] Small 모델 + LoRA (특화)
        print(f"   - Loading 2nd Stage: {ID_SMALL} + LoRA...")
        base_small = WhisperForConditionalGeneration.from_pretrained(
            ID_SMALL, 
            device_map=self.device
        )

        if os.path.exists(ADAPTER_PATH):
            self.stt_model_small_lora = PeftModel.from_pretrained(base_small, ADAPTER_PATH)
            print("     ✅ Small 모델에 LoRA 어댑터 장착 완료!")
        else:
            print(f"     ⚠️ 어댑터 경로 없음. Small 순정으로 동작함.")
            self.stt_model_small_lora = base_small
        
        self.stt_model_small_lora.eval()

        print("✅ 모든 AI 모델 로딩 완료!")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance