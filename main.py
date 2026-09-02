from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag import search_chunks
from question_generation import generate_question
from validate_answer import validate_answer


# Создаём прилку
app = FastAPI(name="ML Interview Coach", version="0.0.1")

# Задаём класс для начальной ручки (тема запроса пользователя)
class StartInterview(BaseModel):
    topic: str = Field(
        min_length=1,
        max_length=200,
        description="Тема для генерации вопроса",
    )

class UserAnswer(BaseModel):
    topic: str = Field(
        min_length=1,
        max_length=200,
    )

    question: str = Field(
        min_length=1,
        max_length=1000
    )

    answer: str = Field(
        min_length=1,
        max_length=5000,
    )

# Ручка с темой запроса пользователя
# Используем пост-запрос, потому что запускаем новую операцию (создание вопроса)
@app.post("/start_interview")
def start_interview(request: StartInterview):
    # Релевантные теме чанки
    materials = search_chunks(request.topic)

    # Генерируем для них вопрос
    question = generate_question(
        topic = request.topic,
        materials=materials,
    )

    materials = search_chunks(request.topic)

    print("\nНайденные материалы:")

    # Возвращаем статус, запрос пользователя и сгенерированный вопрос
    return {
        "status": "success",
        "topic": request.topic,
        "question": question,
    }

@app.post("/user_answer")
def user_answer(request: UserAnswer) -> dict:
    materials = search_chunks(request.topic)

    evaluation = validate_answer(
        topic=request.topic,
        question=request.question,
        user_answer=request.answer,
        materials=materials
    )

    return {
        "status": "success",
        "evaluation": evaluation
    }
