import json
import os

class Router:
    # =================================================
    #  메인 처리 함수
    # =================================================

    def __init__(self, models):
            # 1. 무거운 모델은 외부에서 받아옴 (참조만 함, 메모리 차지 X)
            self.models = models 
            
            # 편의를 위한 바로가기 (Alias)
            self.classifier = models.intent_classifier
            self.chat_model = models.chat_model

            # 2. 사용자별 고유 상태값 (여기는 개별 유지)
            self.waiting_for_decision = False
            self.pending_task = None
            self.history = []

            # Action Map 로드
            self.action_map = {}
            map_path = os.path.join("data", "action_map.json")
            try:
                with open(map_path, "r", encoding="utf-8") as f:
                    self.action_map = json.load(f)
                print(f"[System] Action Map Loaded: {len(self.action_map)} commands")
            except FileNotFoundError:
                print(f"[System] Warning: {map_path} not found.")

    # [전송 함수] Value(긴 코드)를 받아서 전송만 담당
    def _execute_command(self, payload_value: str):
        print(f"🚀 [ROBOT SEND] Sending payload: {payload_value}")
        # 실제 통신 코드 (ROS, HTTP 등) 작성 위치

    def handle(self, text: str):
        # 1) Intent 분류
        intent_result = self.intent_classifier.classify(text)
        intent = intent_result.intent
        print(f"[Intent] {intent} ({intent_result.reason})")

        # 2) 제안 수락 여부 (Decision)
        if self.waiting_for_decision:
            decision = self.decision_model.decide(text)
            if decision == "YES":
                # pending_task는 이미 정확한 Key 값이므로 바로 Map에서 꺼냄
                if self.pending_task in self.action_map:
                    payload = self.action_map[self.pending_task]
                    self._execute_command(payload) # Value 전송
                    response = "알겠습니다. 처리할게요!"
                else:
                    response = "오류가 발생했습니다. 해당 명령 코드를 찾을 수 없어요."
            else:
                response = "알겠습니다. 필요한 게 있을 때 다시 말씀해주세요."

            self.waiting_for_decision = False
            self.pending_task = None
            return response

        # 3) Robot Command 처리 (직접 명령)
        if intent == "robot_command":
            # (1) 번역
            english_text = self.translator.translate(text)
            
            # (2) Normalizer -> Key 획득 (예: "serve_tylenol")
            command_key = self.normalizer.normalize(english_text)
            print(f"[Normalizer Key] {command_key}")

            # (3) Router -> Map Lookup -> Value 획득
            if command_key in self.action_map:
                robot_payload = self.action_map[command_key]
                self._execute_command(robot_payload)
                return "네, 처리할게요."
            else:
                return "죄송해요. 제가 수행할 수 없는 명령이에요."

        # 4) Dialog 처리 (제안 로직 수정됨)
        if intent == "dialog":
            need_action = self.behavior_detector.detect(text)
            
            # (행동 불필요) -> 단순 대화
            if not need_action:
                # ... (기존과 동일)
                context = self.rag.build_context(text)
                if context and context.strip():
                    prompt = (
                        f"사용자 프로필:\n{context}\n\n"
                        f"사용자 입력:\n{text}\n\n"
                        "위 정보를 바탕으로 공감하는 짧은 답변을 하세요."
                    )
                    return self.chat_model.chat(prompt)
                else:
                    return self.chat_model.chat(text)

            # (행동 필요) -> 제안 생성 (Key 포함)
            context = self.rag.build_context(text)
            
            # PersonalResponse가 "멘트 || Key" 형태로 반환함
            generated_output = self.personal_response.generate(text, context)
            
            # [수정 포인트] 따옴표(")까지 확실하게 제거하도록 수정
            if "||" in generated_output:
                suggestion_text, action_key = generated_output.split("||")
                
                # 공백(.strip()) 뿐만 아니라 따옴표(.strip('"'))도 제거
                suggestion_text = suggestion_text.strip().strip('"') 
                action_key = action_key.strip().strip('"')           
            else:
                suggestion_text = generated_output.strip().strip('"')
                action_key = "NONE"

            print(f"[Proposal Log] 멘트: {suggestion_text} / 키: {action_key}")

            # 유효한 Key가 있는 경우에만 대기 상태 진입
            if action_key in self.action_map:
                self.waiting_for_decision = True
                self.pending_task = action_key
                return suggestion_text
            
            else:
                # [수정] 매칭 실패 시 원인을 출력해주는 로그 추가
                if action_key != "NONE":
                    print(f"⚠️ [WARNING] 생성된 Key '{action_key}'가 action_map에 없습니다!")
                    print(f"   (보유 중인 Keys: {list(self.action_map.keys())})")
                
                return suggestion_text

        return "무슨 말씀이신지 잘 이해하지 못했어요."