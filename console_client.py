import json
import requests

BASE_URL = "http://127.0.0.1:8000"

def run_console_client() -> None:
    # Ввод темы вопроса
    topic = input("Введите тему вопроса: ").strip()

    # Дергаем ручку через пост-метод
    question_response = requests.post(
        url=f"{BASE_URL}/start_interview",
        json={
            "topic": topic,
        },
        timeout = (5, 600),
    )

    question_response.raise_for_status()
    question_data = question_response.json()
    session_id = question_data["session_id"]
    question = question_data["question"]

    print(question)

    answer = input("Ваш ответ:")

    answer_response = requests.post(
        url=f"{BASE_URL}/user_answer",
        json={
            "session_id": session_id,
            "answer": answer,
        },
        timeout = (10, 300),
    )

    answer_response.raise_for_status()
    answer_data = answer_response.json()

    print(
        json.dumps(
            answer_data["evaluation"],
            indent=2,
            ensure_ascii=False
        )
    )

if __name__ == "__main__":
    run_console_client()