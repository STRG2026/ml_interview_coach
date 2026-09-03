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
directory = Path("C:/Users/posun/Desktop/ml_interview_coach/data")

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

txt_files = [f for f in directory.glob("*.txt")]

for file in txt_files:
    text = file.read_text(encoding="utf-8")
    file_chunks = text_splitter.create_documents([text]) # file_chunks - объект типа Document, у которого есть поля page_content и metadata

    # Извлекаем lesson_id из имени файла
    lesson_id = file.stem.split("_")[0]  

    # Добавляем метаданные к чанкам
    for chunk_index, chunk in enumerate(file_chunks):
        chunk_id = f"lesson_{lesson_id}_chunk_{chunk_index:03d}"

        chunk_metadata = {
            "source": file.name,
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



