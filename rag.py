import ollama
from typing import TypedDict
from chromadb import PersistentClient

class Material(TypedDict):
    document: str
    metadata: dict
    distance: float

chroma_client = PersistentClient(path="C:\\Users\\posun\\Desktop\\ml_interview_coach\\chroma_data")
collection = chroma_client.get_collection(name="course_lessons")

def search_chunks(topic: str, n_results:int = 3) -> list[Material]:
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
        {
            "document": document,
            "metadata": metadata,
            "distance": distance
        }
        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        )
    ]