import time
import ollama
import uvicorn 
from typing import Literal
from uuid import UUID, uuid4 
from dataclasses import dataclass
from threading import Lock, Thread
from rag import Material, search_chunks
from fastapi import FastAPI, HTTPException
from validate_answer import validate_answer
from validate_answer import EvaluationResult
from console_client import run_console_client
from pydantic import BaseModel, Field, ValidationError
from validate_answer import EvaluationResult, validate_answer
from question_generation import QuestionPackage, generate_question
from generate_reference_answer import ReferenceAnswer, generate_reference_answer


# Создаём прилку
app = FastAPI(title="ML Interview Coach", version="0.0.2")

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

class UserAnswerRequest(BaseModel):
    session_id: UUID
    answer: str = Field(
        min_length=1, 
        max_length=5000
    )

class UserAnswerResponse(BaseModel):
    status: Literal["success"]
    evaluation: EvaluationResult
    reference_answer: str

@dataclass(frozen=True)
class InterviewSession:
    topic: str
    question: QuestionPackage
    materials: list[Material]
    reference_answer: list[Material]

sessions: dict[UUID, InterviewSession] = {}
sessions_lock = Lock()

@app.get("/health")
def health() -> dict[str, str]:
    return {"status" : "ok"}

@app.post("/start_interview", response_model=StartInterviewResponse)
def start_interview(request: StartInterviewRequest) -> StartInterviewResponse:
    topic = request.topic.strip()
    if not topic:
        raise HTTPException(
            status_code=422,
            detail = "Тема не может быть пустой"
        )
    try:
        materials = search_chunks(topic) 
        if not materials:
            raise HTTPException(
                status_code=404,
                detail = "Чанки по этой теме не найдены"
            )

        generated_question = generate_question(
            topic=topic,
            materials=materials
        )

        reference_answer = generate_reference_answer(
            question=generated_question.question,
            materials=materials
        )
        
    except ollama.ResponseError as exp:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama не подготовила интервью: {exp}"
        ) from exp

    except ValidationError as e:
        raise HTTPException(
            status_code=502,
            detail=("Модель вернула вопрос или эталонный ответ в неправильном формате")
        ) from e


    session_id = uuid4()
    session = InterviewSession(
        topic=topic,
        question=generated_question.question,
        reference_answer=reference_answer,
        materials=materials
    )

    with sessions_lock:
        sessions[session_id] = session

    return StartInterviewResponse(
        status="success",
        session_id=session_id, 
        question=session.question
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

    try:
        evaluation = validate_answer(
            question_package = session.question,
            user_answer = answer,
            reference_answer=session.reference_answer,
            materials = session.materials
        )

    except ollama.ResponseError as exp:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama не смогла оценить ответ {exp}"
        ) from exp

    except ValidationError as e:
        raise HTTPException(
            status_code=502,
            detail="Модель вернула оценку в неправильном формате"
        )

    return UserAnswerResponse(
        status="success",
        evaluation=evaluation,
        reference_answer=session.reference_answer.reference_answer
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