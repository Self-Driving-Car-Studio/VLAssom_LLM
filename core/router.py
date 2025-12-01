import json
import os
from core.robot_client import RobotClient

class Router:
    # =================================================
    #  다국어 고정 응답 메시지 정의
    # =================================================
    RESPONSES = {
        "ko": {
            "ok_fetch": "알겠습니다. 바로 가져다드릴게요!",
            "err_code": "오류가 발생했습니다. 해당 명령 코드를 찾을 수 없어요.",
            "ok_standby": "알겠습니다. 필요한 게 있을 때 다시 말씀해주세요.",
            "cmd_ok": "네, 처리할게요.",
            "cmd_cant": "죄송해요. 제가 수행할 수 없는 명령이에요.",
            "unknown": "무슨 말씀이신지 잘 이해하지 못했어요."
        },
        "en": {
            "ok_fetch": "Understood. I'll get that for you right away!",
            "err_code": "An error occurred. I cannot find that command code.",
            "ok_standby": "Understood. Let me know if you need anything else.",
            "cmd_ok": "Okay, I'll handle that.",
            "cmd_cant": "I'm sorry, I cannot perform that command.",
            "unknown": "I didn't quite understand what you meant."
        }
    }

    # =================================================
    #  메인 처리 함수
    # =================================================

    def __init__(self, models):
        # 1. 무거운 모델은 외부에서 받아옴 (참조만 함, 메모리 차지 X)
        self.models = models 
        
        # 편의를 위한 바로가기 (Alias)
        self.classifier = models.intent_classifier
        self.chat_model = models.chat_model
        self.behavior_detector = models.behavior_detector
        self.rag = models.rag
        self.translator = models.translator
        self.normalizer = models.normalizer
        self.personal_response = models.personal_response
        self.decision_model = models.decision_model
        
        # [NEW] 로봇 클라이언트 초기화 (IP는 환경에 맞게 수정, 포트는 8080)
        self.robot_client = RobotClient(host="192.168.0.20", port=8080)

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

    # [전송 함수] Value를 받아서 실제 로봇 서버로 전송
    def _execute_command(self, payload_value: str):
        print(f"✅ [Router] Command successfully sent to Robot Server")
        # 기존 print 대신 RobotClient를 통해 HTTP 요청 전송
        success = self.robot_client.send_task(payload_value)
        
        if success:
            print(f"✅ [Router] Command successfully sent to Robot Server: {payload_value}")
        else:
            print(f"💀 [Router] Failed to send command to Robot Server.")

    def handle(self, text: str):
        # [Step 0] 언어 감지 및 텍스트 정리
        # server.py에서 영어를 요청할 때 덧붙인 태그를 확인
        is_english = False
        if "(Please respond in English)" in text:
            is_english = True
            # 모델 인식을 위해 태그를 제거한 순수 텍스트 추출
            clean_text = text.replace("(Please respond in English)", "").strip()
        else:
            clean_text = text

        # 현재 언어에 맞는 메시지 팩 선택
        lang_key = "en" if is_english else "ko"
        msgs = self.RESPONSES[lang_key]

        # 1) Intent 분류 (태그 제거된 텍스트 사용)
        intent_result = self.classifier.classify(clean_text)
        intent = intent_result.intent
        print(f"[Intent] {intent} ({intent_result.reason}) / English Mode: {is_english}")

        # 2) 제안 수락 여부 (Decision)
        if self.waiting_for_decision:
            # Decision 모델은 "응", "Yes" 등을 처리 (모델이 다국어를 지원한다고 가정)
            decision = self.decision_model.decide(clean_text)
            
            if decision == "YES":
                # pending_task는 이미 정확한 Key 값이므로 바로 Map에서 꺼냄
                if self.pending_task in self.action_map:
                    payload = self.action_map[self.pending_task]
                    self._execute_command(payload) # Value 전송
                    response = msgs["ok_fetch"]
                else:
                    response = msgs["err_code"]
            else:
                response = msgs["ok_standby"]

            self.waiting_for_decision = False
            self.pending_task = None
            return response

        # 3) Robot Command 처리 (직접 명령)
        if intent == "robot_command":
            # (1) 번역 (clean_text 사용)
            # 영어 모드라면 번역기가 영->영 변환을 하거나, 그대로 통과시켜야 함
            english_text = self.translator.translate(clean_text)
            
            # (2) Normalizer -> Key 획득 (예: "serve_tylenol")
            command_key = self.normalizer.normalize(english_text)
            print(f"[Normalizer Key] {command_key}")

            # (3) Router -> Map Lookup -> Value 획득
            if command_key in self.action_map:
                robot_payload = self.action_map[command_key]
                
                # [실행] 로봇 전송
                self._execute_command(robot_payload)
                
                return msgs["cmd_ok"]
            else:
                return msgs["cmd_cant"]

        # 4) Dialog 처리 (제안 로직)
        if intent == "dialog":
            # 현재 언어 설정 확인
            lang_code = "en" if is_english else "ko"
            
            # [Log] 대화 요청 수신 및 언어 확인
            print(f"[Dialog Request] Input: '{clean_text}' | Language: {lang_code}")
            
            need_action = self.behavior_detector.detect(clean_text)
            
            # [Log] 행동 감지 결과
            print(f"[Behavior Detection] Result: {'Action Needed' if need_action else 'Chat Only'}")
            
            # ---------------------------------------------------------
            # (Case A) 행동 불필요 -> 단순 대화 (ChatModel 사용)
            # ---------------------------------------------------------
            if not need_action:
                context = self.rag.build_context(clean_text)
                
                # [Log] RAG가 가져온 문맥 확인
                print(f"[RAG Context - Chat] {context}")
                
                # ChatModel에 전달할 입력 텍스트 구성
                chat_input = clean_text
                if context and context.strip():
                    chat_input = f"User Profile/Context: {context}\n\nUser Input: {clean_text}"
                
                response = self.chat_model.chat(chat_input, lang=lang_code)
                
                # [Log] 최종 응답 기록
                print(f"[Chat Response] Generated: {response}")
                return response

            # ---------------------------------------------------------
            # (Case B) 행동 필요 -> 제안 생성 (PersonalResponse 사용)
            # ---------------------------------------------------------
            context = self.rag.build_context(clean_text)
            
            # [Log] RAG가 가져온 문맥 확인
            print(f"[RAG Context - Action] {context}")
            
            # PersonalResponse 모델 입력 구성
            gen_input_text = clean_text
            if is_english:
                gen_input_text += " (Respond in English)"

            generated_output = self.personal_response.generate(gen_input_text, context, lang_code)
            
            # [Log] LLM 원본 출력 (파싱 전 데이터 확인용)
            print(f"[Raw LLM Output] {generated_output}")
            
            # [안전장치] 파싱 로직 강화 ("멘트 || 키")
            suggestion_text = generated_output
            action_key = "NONE"

            if "||" in generated_output:
                parts = generated_output.split("||")
                if len(parts) >= 2:
                    suggestion_text = parts[0].strip().strip('"') 
                    action_key = parts[1].strip().strip('"')
            else:
                suggestion_text = generated_output.strip().strip('"')

            # [Log] 파싱 결과 확인
            print(f"[Parsed Proposal] Text: '{suggestion_text}' / Key: '{action_key}'")

            # 유효한 Key가 있는 경우에만 대기 상태 진입
            if action_key in self.action_map:
                self.waiting_for_decision = True
                self.pending_task = action_key
                
                # [Log] 유효 키 확인 및 대기 상태 진입
                print(f"[Action Decision] Valid Key '{action_key}'. Entering wait state.")
                
                return suggestion_text, action_key
            
            else:
                if action_key != "NONE":
                    # [Log] 경고: 키는 나왔으나 맵에 없음
                    print(f"⚠️ [WARNING] Invalid Action Key Detected: '{action_key}' (Not in action_map)")
                else:
                    # [Log] 키 없음 (단순 제안 멘트만 생성됨)
                    print("[Action Decision] No actionable key found. Returning text only.")
                
                # 키가 없거나 잘못된 경우 멘트만 반환
                return suggestion_text

        return msgs["unknown"]