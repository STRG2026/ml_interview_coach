import ollama 
from pydantic import BaseModel, Field

class FinalResult(BaseModel):
    final_result: str = Field (
        min_length=1, 
        max_length=5000,
        description="Финальный ответ, который будет выведен пользователю"
    )
    


def generate_feedback(evaluation) -> FinalResult:
    evaluation = evaluation.strip()

    reference_schema = FinalResult.model_json_schema()

    if not evaluation:
        raise ValueError("Функции генерации ответа для пользователя не пришли результаты оценки")

    responce = ollama.chat(
            model = "qwen3.5:9b-q4_K_M",
            messages = [
                {
                    "Ты - ментор для ML-инженеров\n"
                    "Твоя задача - получить на вход JSON-массив, содержащий результат оценки ответа пользователя на вопрос,"
                    "проанализировать этот массив и сгенерировать ответ для пользователя\n"
                    "Правила:"
                    "1. Если поле score имеет значение 0 до 6 - возьми информацию из полей missing_takes и feedback и объясни пользователю его ошибки\n"
                    "2. Если поле score имеет значение от 7 до 8 - похвали пользователя за хороший ответ и если поля missing_takes и feedback не пустые"
                    "возьми информацию из них и укажи пользователю на неточности в ответе"
                    "3. Если поле score имеет значение от 9 до 10 - похвали пользователя за прекрасный ответ и пожелай успехов в узучении машинного обучения \n"
                },
                {
                    "role" : "user",
                    "content" : (
                        f"EVALUATION RESULT: {evaluation}\n"
                    )
                }
            ],
            format = reference_schema,
            # Отключаем thinking, чтобы ускорить инференс. Для этой задачи thinking-mode не нужен
            think = False,
            stream = False,
            options = {
                # Ненулевая температура, так как в этой задаче можем дать LLM небольшую долю свободы
                "temperature" : 0.05,
                "num_ctx" : 4096,
                "num_predict" : 384
            },
            # Оставляем LLM выгруженной чтобы ускорить инференс
            keep_alive = "10m"
        )

    return FinalResult.model_validate_json()
    