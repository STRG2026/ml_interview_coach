import json
import ollama 
from rag import Material
from pydantic import BaseModel, Field

class QuestionPackage(BaseModel):
    question: str = Field(
        min_length=1,
        description="Один вопрос для пользователя"
    )
    reference_answer: str = Field(
        min_length=1,
        description="Эталонный ответ на основе материалов"
    )
    key_points: list[str] = Field(
        min_length=2,
        max_length=50,
        description="Краткие тезисы, которые должны пояснить ответ"
    )


def build_context(materials: list[dict]) -> str:
    context_parts = []

    for index, material in enumerate(materials, start=1):
        if index == 1:
            fragment_type = "ОСНОВНОЙ ФРАГМЕНТ"
        else:
            fragment_type = "ДОПОЛНИТЕЛЬНЫЙ ФРАГМЕНТ"

        context_parts.append(
            f"[{fragment_type} {index}]\n"
            f"{material['document']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return context

def generate_question(topic: str, materials: list[dict]) -> QuestionPackage:
    if not materials:
        raise ValueError("Релевантные чанки не найдены")
    context = build_context(materials)

    question_schema = QuestionPackage.model_json_schema()

    schema_text = json.dumps(
        question_schema,
        ensure_ascii=False
    )

    response = ollama.chat(
        model="qwen3.5:9b-q4_K_M",
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты — ML-инженер и ментор.\n"
                    "Сформируй внутренний пакет для проведения интервью.\n\n"
                    "Заполни все поля:\n"
                    "- question: один вопрос по указанной теме;\n"
                    "- reference_answer: эталонный ответ по материалам;\n"
                    "- key_points: не менее двух ключевых тезисов ответа.\n\n"
                    "Поле question должно содержать только вопрос. "
                    "Не включай в него эталонный ответ или подсказки.\n"
                    "Основной фрагмент имеет наивысший приоритет.\n"
                    "Не используй знания, отсутствующие в материалах.\n"
                    "Верни только JSON без Markdown и пояснений."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"ТЕМА:\n{topic}\n\n"
                    f"МАТЕРИАЛЫ:\n{context}\n\n"
                    f"JSON-СХЕМА:\n{schema_text}"
                ),
            },
        ],
        format=question_schema,
        think=False,
        stream=False,
        options={
            "temperature": 0,
            "num_ctx": 4096,
            "num_predict": 512,
        },
        keep_alive="10m",
    )

    return QuestionPackage.model_validate_json(
        response["message"]["content"]
    )