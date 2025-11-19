from models.intent_classifier import IntentClassifier
from models.translator import Translator
from models.normalizer import Normalizer
from models.chat_model import ChatModel

# 신규 RAG + 행동 판단 모듈들
from models.rag.personal_rag import PersonalRAG
from models.rag.personal_response import PersonalResponse
from models.rag.behavior_detector import BehaviorDetector
from models.rag.decision_model import DecisionModel

# 로봇 모듈 (기존 그대로 유지)
from models.robot_planner import RobotPlanner
from models.robot_car_controller import RobotCarController


class Router:
    def __init__(self):
        # 기존 모듈들
        self.intent_classifier = IntentClassifier()
        self.translator = Translator()
        self.normalizer = Normalizer()
        self.chat_model = ChatModel()

        # 로봇 관련
        self.planner = RobotPlanner()
        self.robot_car = RobotCarController()

        # 신규 AI 모듈들
        self.rag = PersonalRAG()
        self.personal_response = PersonalResponse()
        self.behavior_detector = BehaviorDetector()
        self.decision_model = DecisionModel()

        # 제안 후 응답 상태
        self.waiting_for_decision = False
        self.pending_task = None   # normalized single_task 저장용


    # =================================================
    #  메인 처리 함수
    # =================================================
    def handle(self, text: str):
        # 1) Intent 분류
        intent_result = self.intent_classifier.classify(text)
        intent = intent_result.intent
        print(f"[Intent] {intent} ({intent_result.reason})")

        # -------------------------------------------------------
        # 2) 제안 → 사용자 응답 YES/NO 판단 state인지 체크
        # -------------------------------------------------------
        if self.waiting_for_decision:
            decision = self.decision_model.decide(text)
            print(f"[Decision] {decision}")

            if decision == "YES":
                print(f"[Robot] Task 실행: {self.pending_task}")

                plan = self.planner.plan(self.pending_task)
                if plan:
                    linear = plan["linear"]
                    angular = plan["angular"]
                    duration = plan["duration"]

                    # 로봇카 제어 (기존 방식 유지)
                    self.robot_car.send_speed(linear, angular)
                    import time
                    time.sleep(duration)
                    self.robot_car.send_speed(0, 0)

                response = "도와드릴게요!"
            else:
                response = "알겠습니다. 필요한 게 있을 때 다시 말씀해주세요."

            self.waiting_for_decision = False
            self.pending_task = None
            return response

        # =======================================================
        # 3) robot_command (기존 로직 유지)
        # =======================================================
        if intent == "robot_command":
            english = self.translator.translate(text)
            command = self.normalizer.normalize(english)

            print(f"[Translate] {english}")
            print(f"[Normalize] {command}")

            plan = self.planner.plan(command)
            if plan:
                linear = plan["linear"]
                angular = plan["angular"]
                duration = plan["duration"]

                self.robot_car.send_speed(linear, angular)
                import time
                time.sleep(duration)
                self.robot_car.send_speed(0, 0)

            return f"명령을 수행합니다: {command}"

        # =======================================================
        # 4) dialog → 행동 필요 여부 판단
        # =======================================================
        if intent == "dialog":
            need_action = self.behavior_detector.detect(text)
            print(f"[BehaviorNeeded] {need_action}")

            # --------- 행동 필요 없음 → 일반 대화 ---------
            if not need_action:
                return self.chat_model.chat(text)

            # --------- 행동 필요 → RAG 기반 제안 생성 ---------
            context = self.rag.build_context(text)
            suggestion = self.personal_response.generate(text, context)

            print(f"[Suggestion] {suggestion}")

            # suggestion을 영어로 번역 → canonical task로 변환
            english = self.translator.translate(suggestion)
            canonical = self.normalizer.normalize(english)

            print(f"[Translator->Canonical] {canonical}")

            # 다음 사용자 답변에서 YES/NO 판단하도록 대기
            self.waiting_for_decision = True
            self.pending_task = canonical

            return suggestion

        # =======================================================
        # 5) medicine_info (기존 기능 나중에)
        # =======================================================
        if intent == "medicine_info":
            return "약 정보 기능은 아직 준비 중입니다."

        # =======================================================
        # 6) fallback
        # =======================================================
        return "무슨 말씀이신지 잘 이해하지 못했어요."