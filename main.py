import time
import uvicorn 
from typing import Literal
from uuid import UUID, uuid4 
from dataclasses import dataclass
from threading import Lock, Thread
from pydantic import BaseModel, Field 
from rag import Material, search_chunks
from fastapi import FastAPI, HTTPException
from validate_answer import validate_answer
from validate_answer import EvaluationResult
from console_client import run_console_client
from question_generation import QuestionPackage, generate_question

# Создаём прилку
app = FastAPI(title="ML Interview Coach", version="0.0.1")

# Задаём класс для начальной ручки (тема запроса пользователя)
class StartInterviewRequest(BaseModel):
    topic: str = Field(
        min_length=1,
        max_length=200,
        description="Тема для генерации вопроса",
    )

class StartInterviewResponse(BaseModel):
    status: Literal["success"]
    session_id: UUID
    question: str
    materials: list[dict]

class UserAnswerRequest(BaseModel):
    session_id: UUID
    answer: str = Field(
        min_length=1, 
        max_length=5000
    )

class UserAnswerResponse(BaseModel):
    status: Literal["success"]
    evaluation: EvaluationResult

@dataclass(frozen=True)
class InterviewSession:
    topic: str
    question_package: QuestionPackage
    materials: list[Material]

sessions: dict[UUID, InterviewSession] = {}
sessions_lock = Lock()

@app.post("/start_interview", response_model=StartInterviewResponse)
def start_interview(request: StartInterviewRequest) -> StartInterviewResponse:
    topic = request.topic.strip()
    if not topic:
        raise HTTPException(
            status_code=422,
            detail = "Тема не может быть пустой"
        )

    materials = search_chunks(topic) 
    if not materials:
        raise HTTPException(
            status_code=404,
            detail = "Чанки по этой теме не найдены"
        )

    question_package = generate_question(
        topic=topic,
        materials=materials
    )

    session_id = uuid4()
    session = InterviewSession(
        topic=topic,
        question_package=question_package,
        materials=materials
    )

    with sessions_lock:
        sessions[session_id] = session

    return StartInterviewResponse(
        status="success",
        session_id=session_id, 
        question=question_package.question,
        materials=materials  
    )

@app.post("/user_answer", response_model=UserAnswerResponse)
def user_answer(request: UserAnswerRequest) -> UserAnswerResponse:
    answer = request.answer.strip()
    if not answer:
        raise HTTPException(
            status_code=422,
            detail="Ответ не может быть пустым"
        )
    with sessions_lock:
        session = sessions.get(request.session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Сессия не найдена или сервер не запущен"
        )


    evaluation = validate_answer(
        question_package = session.question_package,
        user_answer = answer,
        materials = session.materials
    )

    return UserAnswerResponse(
        status="success",
        evaluation=evaluation
    )

def run_project() -> None:
    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )

    server = uvicorn.Server(config=config)

    server_thread = Thread(
        target=server.run,
        name="uvicorn_server",
        daemon=True
    )
    server_thread.start()

    startup_deadline = time.monotonic() + 10

    while not server.started:
        if not server_thread.is_alive():
            raise RuntimeError(
                "FastAPI не поднялся, возможно занят 8000-ый порт"
            )

        if time.monotonic() >= startup_deadline:
            server.should_exit = True
            server_thread.join(timeout=5)
            raise TimeoutError("FastAPI не поднялся за 10 секунд")

        time.sleep(0.05)

    try:
        run_console_client()
    finally:
        server.should_exit = True
        server_thread.join(timeout=5)

if __name__ == "__main__":
    run_project()