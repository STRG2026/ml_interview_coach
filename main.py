import time
import ollama
import uvicorn 
from typing import Literal
from uuid import UUID, uuid4 
from dataclasses import dataclass
from threading import Lock, Thread
from rag import Material, search_chunks
from fastapi import FastAPI, HTTPException
from console_client import run_console_client
from generate_feedback import generate_feedback
from pydantic import BaseModel, Field, ValidationError
from validate_answer import EvaluationResult, validate_answer
from question_generation import QuestionPackage, generate_question
from generate_reference_answer import ReferenceAnswer, generate_reference_answer


# Создаём прилку
app = FastAPI(title="ML Interview Coach", version="0.0.3")

# Задаём класс для начальной ручки (тема запроса пользователя)
class StartInterviewRequest(BaseModel):
    topic: str = Field(
        min_length=1,
        max_length=200,
        description="Тема для генерации вопроса"
    )

class StartInterviewResponse(BaseModel):
    status: Literal["success"]
    session_id: UUID
    question: str

class UserAnswerRequest(BaseModel):
    session_id: UUID
    answer: str = Field(
        min_length=1, 
        max_length=5000,
        description="Ответ пользователя на вопрос"
    )

class UserAnswerResponse(BaseModel):
    status: Literal["success"]
    evaluation: EvaluationResult
    reference_answer: str
    final_feedback: str

@dataclass(frozen=True)
class InterviewSession:
    topic: str
    question_package: QuestionPackage
    materials: list[Material]
    reference_answer: ReferenceAnswer
    created_at: float

sessions: dict[UUID, InterviewSession] = {}
# Защищаем словаь сессии
sessions_lock = Lock()
# Не допускаем нагрузку LLM несколькими запросами
llm_lock = Lock()

def get_session(session_id: UUID) -> InterviewSession | None:
    with sessions_lock:
        session = sessions.get(session_id)

        if session is None:
            return None

        session_age = (
            time.monotonic() - session.created_at
        )

        if session_age > 3600:
            sessions.pop(session_id, None)
            return None

        return session

@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
    }


@app.post("/start_interview", response_model=StartInterviewResponse)
def start_interview(request: StartInterviewRequest) -> StartInterviewResponse:
    topic = request.topic.strip()

    if not topic:
        raise HTTPException(
            status_code=422,
            detail="Тема не может быть пустой"
        )

    try:
        with llm_lock:
            materials = [
            material
            for material in search_chunks(topic)
            if material.distance <= 0.55
        ]

            if not materials:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        "Материалы по теме не найдены, дополни RAG или проверь релевантность вопроса"
                    ),
                )

            question_package = generate_question(
                topic=topic,
                materials=materials
            )

            reference_answer = generate_reference_answer(
                question=question_package.question,
                materials=materials
            )

    except HTTPException:
        raise

    except ollama.ResponseError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "LLM не вернула данные, ошибка: "
                f"{error}"
            )
        ) from error

    except ConnectionError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Не удалось подключиться к Ollama, введи Ollama ps"
            )
        ) from error

    except ValidationError as error:
        raise HTTPException(
            status_code=502,
            detail=(
                "Модель вернула вопрос или эталонный "
                "ответ в неправильном формате"
            )
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=502,
            detail=str(error)
        ) from error

    session_id = uuid4()

    interview_session = InterviewSession(
        topic=topic,
        question_package=question_package,
        materials=materials,
        reference_answer=reference_answer,
        created_at=time.monotonic()
    )

    with sessions_lock:
        sessions[session_id] = interview_session

    return StartInterviewResponse(
        status="success",
        session_id=session_id,
        question=question_package.question
    )


@app.post("/user_answer", response_model=UserAnswerResponse)
def user_answer(request: UserAnswerRequest) -> UserAnswerResponse:
    answer = request.answer.strip()

    if not answer:
        raise HTTPException(
            status_code=422,
            detail="Ответ не может быть пустым"
        )

    session = get_session(request.session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Сессия не найдена или срок её действия истёк"
            )
        )

    try:
        with llm_lock:
            try:
                evaluation = validate_answer(
                    question=session.question_package.question,
                    user_answer=answer,
                    reference_answer=session.reference_answer,
                    materials=session.materials,
                )

                evaluation = EvaluationResult.model_validate(
                    evaluation
                )

            except ValidationError as error:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Этап оценки ответа вернул "
                        f"невалидный JSON: {error}"
                    ),
                ) from error

            except RuntimeError as error:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Ошибка на этапе оценки ответа: "
                        f"{error}"
                    ),
                ) from error
            
            try:
                final_feedback = generate_feedback(
                    evaluation=evaluation,
                    reference_answer=session.reference_answer,
                )

            except ValidationError as error:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Этап генерации финального feedback "
                        f"вернул невалидный JSON: {error}"
                    ),
                ) from error

            except RuntimeError as error:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Ошибка на этапе генерации feedback: "
                        f"{error}"
                    ),
                ) from error

    except HTTPException:
        raise

    except ollama.ResponseError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama не смогла обработать ответ: "
                f"{error}"
            ),
        ) from error

    except ConnectionError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Не удалось подключиться к Ollama. "
                "Проверь состояние командой `ollama ps`."
            ),
        ) from error

    with sessions_lock:
        sessions.pop(request.session_id, None)

    return UserAnswerResponse(
        status="success",
        evaluation=evaluation,
        reference_answer=(session.reference_answer.reference_answer),
        final_feedback=final_feedback.final_result
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
                "FastAPI не запустился, возможно занят 8000 порт"
            )

        if time.monotonic() >= startup_deadline:
            server.should_exit = True
            server_thread.join(timeout=5)

            raise TimeoutError(
                "FastAPI не запустился за 10 секунд"
            )

        time.sleep(0.05)

    try:
        run_console_client()
    finally:
        server.should_exit = True
        server_thread.join(timeout=10)

        if server_thread.is_alive():
            print(
                "FastAPI не успел завершиться за 10 секунд"
            )


if __name__ == "__main__":
    run_project()