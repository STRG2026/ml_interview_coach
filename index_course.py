import ollama
from pathlib import Path
from chromadb import PersistentClient
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Создали клиент (аналог БД в PGSQL)
chroma_client = PersistentClient(path="C:\\Users\\posun\\Desktop\\ml_interview_coach\\chroma_data")

# Создали коллекцию, общую для всех уроков курса (аналог таблицы в PGSQL)
collection = chroma_client.get_or_create_collection(
    name="course_lessons",
    configuration={
        "hnsw": {
            "space": "cosine"
        }
    }
)

# Массив конспектов лекций
file_paths = [Path("data/001_chto_takoe_mashinnoe_obuchenie_obuchayushchaya_vyborka.txt"),
              Path("data/002_postanovka_zadachi_mashinnogo_obuchenia.txt")]

# Задаём параметры сплиттера
text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1400,
            chunk_overlap = 150, 
            keep_separator=False,
            strip_whitespace=True,
        )

# Нарезаем чанки
ids = []
documents = []
metadatas = []

for file_path in file_paths:
    text = file_path.read_text(encoding="utf-8")
    file_chunks = text_splitter.create_documents([text]) # file_chunks - объект типа Document, у которого есть поля page_content и metadata

    # Извлекаем lesson_id из имени файла
    lesson_id = file_path.stem.split("_")[0]  

    # Добавляем метаданные к чанкам
    for chunk_index, chunk in enumerate(file_chunks):
        chunk_id = f"lesson_{lesson_id}_chunk_{chunk_index:03d}"

        chunk_metadata = {
            "source": file_path.name,
            "lesson_id": lesson_id,
            "chunk_index": chunk_index
        }

        ids.append(chunk_id)
        documents.append(chunk.page_content)
        metadatas.append(chunk_metadata)

# Создаём ембеддинги
batch = ollama.embed(
    model='qwen3-embedding:0.6b',
    input = documents
)

embeddings = batch["embeddings"]

# Добавляем ембеддинги в коллецию Chroma
# Upsert автоматически добавит новые или перезапишет старые записи
collection.upsert(
    ids=ids,
    embeddings=embeddings,
    metadatas=metadatas,
    documents=documents
)



