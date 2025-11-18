from core.router import Router


class Pipeline:
    """
    상위 레벨 파이프라인 — Router를 사용해 전체 흐름을 통합 관리.
    """

    def __init__(self):
        self.router = Router()

    def run(self, user_input: str) -> str:
        """
        사용자가 입력한 문장을 받아 Router에 처리를 위임하고 결과를 반환.
        """
        result = self.router.handle(user_input)
        return result