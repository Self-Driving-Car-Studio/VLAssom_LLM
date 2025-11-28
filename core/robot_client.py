import requests
import json

class RobotClient:
    def __init__(self, host="192.168.0.20", port=8080):
        # 로봇 추론 서버 주소 및 포트 설정 (요청하신 8080 포트 사용)
        self.base_url = f"http://{host}:{port}"
        self.headers = {"Content-Type": "application/json"}
    
    def send_task(self, task_description: str) -> bool:
        """
        로봇 추론 서버(FastAPI)로 자연어 태스크 명령을 전송합니다.
        Endpoint: POST /run_task
        """
        endpoint = "/run_task"
        url = f"{self.base_url}{endpoint}"
        payload = {"task": task_description}
        
        try:
            print(f"🚀 [RobotClient] Sending Task to {url} | Task: {task_description}")
            
            # 타임아웃 10초 설정 (네트워크 지연 고려)
            response = requests.post(url, json=payload, headers=self.headers, timeout=10)
            
            # HTTP 200 OK 확인
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    print(f"✅ [RobotClient] Success: {data}")
                    return True
                else:
                    print(f"⚠️ [RobotClient] Server responded with error: {data}")
                    return False
            else:
                print(f"❌ [RobotClient] HTTP Failed: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"⚠️ [RobotClient] Connection Refused: 로봇 서버({self.base_url})가 켜져 있는지 확인하세요.")
            return False
        except requests.exceptions.RequestException as e:
            print(f"⚠️ [RobotClient] Error: {e}")
            return False