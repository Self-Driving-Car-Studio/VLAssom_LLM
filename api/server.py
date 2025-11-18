from fastapi import FastAPI
from pydantic import BaseModel
from core.pipeline import Pipeline

app = FastAPI()
pipeline = Pipeline()

class UserInput(BaseModel):
    text: str

@app.post("/api/assistant")
def assistant_api(payload: UserInput):
    user_text = payload.text
    result = pipeline.run(user_text)

    return {
        #"input": user_text,
        "result": result
    }
