import ollama 

def build_context(materials: list[dict]) -> str:
    context_parts = []

    for index, material in enumerate(materials, start=1):
        if index == 1:
            fragment_type = "ОСНОВНОЙ ФРАГМЕНТ"
        else:
            fragment_type = "ДОПОЛНИТЕЛЬНЫЙ ФРАГМЕНТ"

        context_parts.append(
            f"[{fragment_type} {index}]\n"
            f"{material['documents']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return context

def generate_question(topic: str, materials: list[dict]) -> str:
    context = build_context(materials)

    response = ollama.chat(
        model = 'qwen2.5:7b',
        messages = [{
                "role": "system",
                "content": (
                    "Ты - ML-инженер и ментор "
                    "Твоя задача - сформулировать один вопрос, непосредственно проверяющий понимание "
                    "указанной темы.\n\n"
                    "Правила:\n"
                    "1. Главным предметом вопроса должна быть указанная тема.\n"
                    "2. Основной фрагмент имеет наивысший приоритет.\n"
                    "3. Дополнительные фрагменты используй только тогда, когда они "
                    "непосредственно раскрывают указанную тему.\n"
                    "4. Игнорируй сведения о соседних понятиях, даже если из них проще "
                    "составить вопрос.\n"
                    "5. Если тема является понятием, проверь понимание его определения, "
                    "состава, назначения или роли.\n"
                    "6. Вопрос должен быть понятен без исходных материалов.\n"
                    "7. Не сообщай правильный ответ.\n"
                    "8. Выведи только один вопрос.\n\n"
                    "Перед выводом проверь: ожидаемый ответ кандидата должен в первую "
                    "очередь объяснять указанную тему."
                ),
            },
            {
                "role" : "user",
                "content": (
                    f"Тема вопроса: {topic}\n\n"
                    f"Материалы курса: \n{context}"
                ),
            },
        ],
        options = {
            "temperature": 0,
            "num_ctx": 8192,
            "num_predict": 512,
        },
        keep_alive=0,
    )

    question = response["message"]["content"]

    return question