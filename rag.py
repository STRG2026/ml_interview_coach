import ollama
from chromadb import PersistentClient
from typing import List

chroma_client = PersistentClient(path="C:\\Users\\posun\\Desktop\\ml_interview_coach\\chroma_data")
collection = chroma_client.get_collection(name="course_lessons")

def search_chunks(topic: str, n_results:int = 3) -> List[dict]:
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
        query_embeddings=query_embeddings,
        n_results=3,
        include = ["documents", "metadatas", "distances"]
    )

    # Распаковываем метаданные
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    # Собираем инфу по чанкам воедино
    materials = []

    for doc, meta, dist in zip(documents, metadatas, distances):
        materials.append({
            "documents": doc,
            "metadatas": meta,
            "distances": dist
        })

    return materials