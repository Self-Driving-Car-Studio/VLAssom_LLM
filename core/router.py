from models.intent_classifier import IntentClassifier
from models.translator import Translator
from models.normalizer import Normalizer
from models.chat_model import ChatModel
from models.robot_car_controller import RobotCarController
from models.robot_planner import RobotPlanner
import json
import os
import time

class Router:
    """
    Intent에 따라 적절한 처리 경로로 분기하는 Router.
    """

    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.translator = Translator()
        self.normalizer = Normalizer()
        self.chat_model = ChatModel()
        # rag_retriever는 이후 단계에서 추가 예정

        # 로봇카 제어기 연결
        self.robot_car = RobotCarController()
        self.planner = RobotPlanner()

        # drive_map.json 로드
        drive_map_path = "./data/planner_map.json"
        with open(drive_map_path, "r") as f:
            self.drive_map = json.load(f)

    def handle(self, text: str) -> str:
        """
        입력 텍스트를 intent에 따라 적절한 모듈에 전달하고 결과 문자열을 반환.
        """
        intent_result = self.intent_classifier.classify(text)
        intent = intent_result.intent

        #print(f"[Intent] {intent} ({intent_result.reason})")

        # 1) 로봇 명령일 경우
        if intent == "robot_command":
            # 1-1) 번역
            english = self.translator.translate(text)
            #print(f"[Translate] {english}")

            # 1-2) 정규화
            task = self.normalizer.normalize(english)
            #print(f"[Normalize] {task}")

            # If task is a robot car command
            if task in self.drive_map:
                linear = self.drive_map[task]["linear"]
                angular = self.drive_map[task]["angular"]

                ack = self.robot_car.send_speed(linear, angular)
                print(f"[RobotCar] Sent speeds → {ack}")

                return f"{task}"
            # 2) robot arm command → planner 확인
            else:
                planner_cmd = self.planner.plan(task)
                if planner_cmd:
                    linear = planner_cmd["linear"]
                    angular = planner_cmd["angular"]
                    duration = planner_cmd["duration"]

                    print(f"[RobotCar][Planner] {task} → {linear}/{angular} for {duration} sec")

                    if duration > 0:
                        self.robot_car.send_speed(linear, angular)
                        time.sleep(duration)
                        self.robot_car.send_speed(0, 0)

                    return task

        # 2) 약 정보 intent → (다음 단계에서 구현)
        elif intent == "medicine_info":
            return "(TODO) 약 정보 모듈 연결 예정"

        # 3) 일반 대화 intent → (다음 단계에서 구현)
        else:
            response = self.chat_model.chat(text)
            #print(f"[Chat] {response}")
            return response