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
            #print(f"[Decision] {decision}")

            if decision == "YES":
                #print(f"[Robot] Task 실행: {self.pending_task}")

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

            #print(f"[Translate] {english}")
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
            #print(f"[BehaviorNeeded] {need_action}")

            # --------- 행동 필요 없음 → RAG-enhanced 일반 대화 ---------
            if not need_action:
                # 개인 프로필 기반 context 생성
                context = self.rag.build_context(text)
                #print(f"[RAG Context]\n{context}\n")

                # context가 있으면 개인화 프롬프트, 없으면 기존 그대로
                if context and context.strip():
                    prompt = (
                        "아래는 사용자의 개인 프로필 정보입니다. 이 정보를 참고하되, "
                        "사실처럼 단정 짓지 말고 '추측'임을 전제로 공감과 조언 위주로 답변하세요.\n"
                        "프로필이 현재 대화와 크게 상관 없으면 무시해도 됩니다.\n\n"
                        f"=== 사용자 프로필 ===\n{context}\n"
                        "=====================\n\n"
                        f"=== 사용자 입력 ===\n{text}\n"
                        "=====================\n\n"
                        "위 정보를 모두 고려해서, 자연스러운 한국어 한두 문장으로 대답하세요."
                    )
                    return self.chat_model.chat(prompt)
                else:
                    # RAG가 아무 것도 못 찾으면 예전처럼 동작
                    return self.chat_model.chat(text)

            # --------- 행동 필요 → RAG 기반 제안 생성 (기존 로직 유지) ---------
            context = self.rag.build_context(text)
            suggestion = self.personal_response.generate(text, context)

            #print(f"[Suggestion] {suggestion}")

            # suggestion을 영어로 번역 → canonical task로 변환
            english = self.translator.translate(suggestion)
            canonical = self.normalizer.normalize(english)

            #print(f"[Translator->Canonical] {canonical}")

            # 다음 사용자 답변에서 YES/NO 판단하도록 대기
            self.waiting_for_decision = True
            self.pending_task = canonical

            return suggestion

        # =======================================================
        # 6) fallback
        # =======================================================
        return "무슨 말씀이신지 잘 이해하지 못했어요."