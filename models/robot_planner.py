import json

class RobotPlanner:
    def __init__(self, path="./data/planner_map.json"):
        with open(path, "r") as f:
            self.map = json.load(f)

    def plan(self, task: str):
        # 해당 task가 planner_map.json에 있으면 그대로 반환
        if task in self.map:
            return self.map[task]
        
        return None