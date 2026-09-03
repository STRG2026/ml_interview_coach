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
                    "You are an ML engineer and mentor.\n"
                    "Create an internal package for conducting an interview\n"
                    "Fill in fields:\n"
                    "- question: one question on the specified topic\n"
                    "- reference_answer: a reference answer based on the materials\n"
                    "- key_points: at least two key points of the answer\n"
                    "The question fields should contain only the question\n"
                    "Do not include a reference answer of hints in it\n"
                    "The main fragment has the highest priority\n"
                    "Do not use knowledge that is not present in the materials\n"
                    "Return only JSON without Markdown and explanations\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"TOPIC:\n{topic}\n\n"
                    f"MATERIALS:\n{context}\n\n"
                    f"JSON-SCHEMA:\n{schema_text}"
                ),
            },
        ],
        format=question_schema,
        think=False,
        stream=False,
        options={
            "temperature": 0.4,
            "num_ctx": 4096,
            "num_predict": 512,
        },
        keep_alive="10m",
    )

    return QuestionPackage.model_validate_json(
        response["message"]["content"]
    )