import json
import difflib
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class Normalizer:
    """
    영어 명령을 로봇이 이해하는 single_task 문장으로 변환.
    1) action_map에서 exact match 검사
    2) 없다면 LLM으로 유사도 기반 변환
    """

    def __init__(self,
                 map_path="./data/action_map.json",
                 model_name="Qwen/Qwen2.5-3B-Instruct"):

        print("[Normalizer] 로딩 중...")

        # 룰 기반 매핑 로드
        with open(map_path, "r") as f:
            self.action_map = json.load(f)

        # LLM 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        print("[Normalizer] 로딩 완료")

    def normalize(self, english_text: str) -> str:
        input_text = english_text.lower().strip()

        # 1) 룰 기반 exact match
        if input_text in self.action_map:
            return self.action_map[input_text]

        # 2) 룰 기반 유사 문자열 찾기 (optional)
        close = difflib.get_close_matches(input_text, self.action_map.keys(), n=1, cutoff=0.6)
        if close:
            return self.action_map[close[0]]

        # 3) LLM fallback
        return self.llm_normalize(input_text)

    def llm_normalize(self, text: str) -> str:
        """
        LLM을 사용하여 text를 가장 적절한 single_task로 정규화.
        """

        prompt = f"""
                You are a robot command normalizer.
                Your job is to convert English natural language commands into a single standardized task string.

                Available tasks:
                {json.dumps(list(self.action_map.values()), indent=2)}

                Rules:
                - Return ONLY one of the task strings above.
                - No explanation.

                User command: "{text}"
                Normalized task:
                """

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.0,
            do_sample=False
        )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 마지막 답변만 추출
        last_line = result.split("Normalized task:")[-1].strip()

        return last_line