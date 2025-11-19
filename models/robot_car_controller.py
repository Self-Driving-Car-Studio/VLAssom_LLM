import zmq
import json

class RobotCarController:
    """
    ZMQ를 통해 로봇카로 linear / angular 속도 명령을 보내는 클라이언트.
    """

    def __init__(self, ip="192.168.0.3", port=5556):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{ip}:{port}")
        print(f"[RobotCarController] Connected to tcp://{ip}:{port}")

    def send_speed(self, linear: float, angular: float) -> str:
        msg = json.dumps({"linear": linear, "angular": angular})
        self.socket.send_string(msg)
        ack = self.socket.recv_string()
        return ack