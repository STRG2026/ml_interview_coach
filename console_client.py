import json
import requests
from typing import Any

def read_non_empty(prompt: str) -> str:
    while True:
        value = input(prompt).strip()

        if value:
            return value

        print("Значение не должно быть пустым.")


def extract_api_error(response: requests.Response) -> str:
    try:
        response_data = response.json()
    except ValueError:
        return response.text or "Сервер не сообщил подробности ошибки"

    detail = response_data.get("detail", response_data)

    if isinstance(detail, str):
        return detail

    return json.dumps(
        detail,
        ensure_ascii=False,
        indent=2
    )


def post_json(session: requests.Session, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = session.post(
            url=f"http://127.0.0.1:8000 {endpoint}",
            json=payload,
            timeout=(10, 600)
        )

    response_data = response.json()

    return response_data


def run_console_client() -> None:
    try:
        topic = read_non_empty("Введите тему вопроса: ")

        with requests.Session() as session:
            question_data = post_json(
                session=session,
                endpoint="/start_interview",
                payload={
                    "topic": topic
                },
            )

            session_id = question_data.get("session_id")
            question = question_data.get("question")

            print(f"\nВопрос:\n{question}")

            user_answer = read_non_empty("\nВаш ответ: ")

            answer_data = post_json(
                session=session,
                endpoint="/user_answer",
                payload={
                    "session_id": session_id,
                    "answer": user_answer,
                }
            )

        evaluation = answer_data.get("evaluation")
        final_feedback = answer_data.get("final_feedback")
        reference_answer = answer_data.get("reference_answer")

        score = evaluation.get("score")
        verdict = evaluation.get("verdict")

        print("\nРезультат:")

        if score is not None:
            print(f"Оценка: {score}/10")

        if verdict:
            print(f"Вердикт: {verdict}")

        print(f"\nКомментарий ментора:\n{final_feedback}")

        if isinstance(reference_answer, str) and reference_answer:
            print(f"\nЭталонный ответ:\n{reference_answer}")

    except (KeyboardInterrupt, EOFError):
        print("\nПользователь остановил работу консоли")


if __name__ == "__main__":
    run_console_client()