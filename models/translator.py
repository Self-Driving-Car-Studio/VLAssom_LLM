from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Translator:
    """
    한국어 → 영어 번역기
    robot_command일 때만 사용.
    """

    def __init__(self, model_name="Helsinki-NLP/opus-mt-ko-en"):
        print("[Translator] 모델 로딩 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("[Translator] 로딩 완료")

    def translate(self, text: str) -> str:
        """
        한국어 문장을 영어로 번역하고 후처리까지 수행.
        """
        if not text or not text.strip():
            return ""

        # 1) 번역
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=200,
            num_beams=5,
            early_stopping=True
        )

        raw_english = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 2) 후처리
        cleaned = self.postprocess(raw_english)
        return cleaned

    def postprocess(self, text: str) -> str:
        text = text.lower().strip()

        # 불필요한 공손 표현 제거
        text = text.replace("please", "").strip()

        # "get me" → "bring"
        if text.startswith("get me"):
            text = text.replace("get me", "bring", 1)

        # 위치 부사 제거
        drops = ["over there", "there", "over here", "here"]
        for d in drops:
            text = text.replace(d, "").strip()

        # "that red ball" → "the red ball"
        text = text.replace("that ", "the ")

        # "drawer door" → "drawer"
        text = text.replace("drawer door", "drawer")

        # 문장 뒤의 불필요한 문장부호 제거
        while text.endswith((".", ",", "!", "?")):
            text = text[:-1].strip()

        return text

