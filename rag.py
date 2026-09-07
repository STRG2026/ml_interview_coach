import ollama
from pathlib import Path
from typing import TypedDict, Any
from chromadb import PersistentClient


class Material(BaseModel):
    document: str
    metadata: dict[str, Any]
    distance: float

PROJECT_ROOT = Path(__file__).resolve().parent

chroma_client = PersistentClient(path=str(PROJECT_ROOT / "chroma_data"))
collection = chroma_client.get_collection(name="course_lessons")

def search_chunks(topic: str, n_results:int = 3) -> list[Material]:
    topic = topic.strip()

    if not topic:
        raise ValueError("Тема для поиска не может быть пустой")

    if n_results <= 0:
        raise ValueError("n_results должен быть больше нуля")
    # Вычисляем ембеддинг запроса и выполняем поиск по нему, чтобы использовать везде одну модель
    query_batch = ollama.embed(
        model='qwen3-embedding:0.6b',
        input=topic,
        keep_alive=0
    )

    # Распаковываем вектор, потому-что query_batch сейчас вложенный список
    query_embeddings = query_batch["embeddings"][0]

    # Формируем запрос к Chroma
    result = collection.query(
        query_embeddings=[query_embeddings],
        n_results=n_results,
        include = ["documents", "metadatas", "distances"]
    )

    # Распаковываем метаданные
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    # Собираем инфу по чанкам воедино
    return [
        Material(
            document=document,
            metadata=metadata or {},
            distance=float(distance),
        )
        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        )
        if document is not None and distance is not None
    ]