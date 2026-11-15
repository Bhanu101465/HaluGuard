from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from .pipeline import run

app = FastAPI(title="HaluGuard")

class Req(BaseModel):
    question: str
    passages: list[str]
    given_answer: Optional[str] = None

@app.post("/check")
def check(req: Req):
    return run(req.question, req.passages, given_answer=req.given_answer)
