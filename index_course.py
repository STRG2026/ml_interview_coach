import ollama
from pathlib import Path
from chromadb import PersistentClient
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
CHROMA_DIRECTORY = PROJECT_ROOT / "chroma_data"

def get_text_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Директория с материалами не найдена: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Путь не является директорией: {directory}"
        )

    # Сортируем для воспроизводимого порядка индексации
    text_files = sorted(directory.glob("*.txt"))

    if not text_files:
        raise FileNotFoundError(
            f"В директории {directory} нет нужных файлов"
        )

    return text_files


def prepare_documents(text_files: list[Path]) -> tuple[list[str], list[str], list[dict[str, str | int]]]:

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1400,
        chunk_overlap=150,
        keep_separator=False,
        strip_whitespace=True
    )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int]] = []

    for file_path in text_files:
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            print(f"Пропущен пустой файл: {file_path.name}")
            continue

        # create_documents() возвращает список объектов Document
        file_chunks = text_splitter.create_documents([text])

        lesson_id = file_path.stem.split("_", maxsplit=1)[0]

        if not lesson_id:
            raise ValueError(
                f"Не удалось извлечь lesson_id из файла {file_path.name}"
            )

        for chunk_index, chunk in enumerate(file_chunks):
            document = chunk.page_content.strip()

            if not document:
                continue

            # Полное имя файла - уникальный айдишник
            chunk_id = (
                f"{file_path.stem}_chunk_{chunk_index:03d}"
            )

            # Cобираем метаданные чанков воедино
            chunk_metadata: dict[str, str | int] = {
                "source": file_path.name,
                "lesson_id": lesson_id,
                "chunk_index": chunk_index
            }

            ids.append(chunk_id)
            documents.append(document)
            metadatas.append(chunk_metadata)

        print(
            f"Обработан файл {file_path.name}: "
            f"{len(file_chunks)} чанков"
        )

    if not documents:
        raise ValueError(
            "После обработки файлов не было создано ни одного чанка"
        )

    if not (len(ids) == len(documents) == len(metadatas)):
        raise RuntimeError(
            "Количество айдишников, документов и метаданных не совпадает"
        )

    return ids, documents, metadatas


def create_embeddings(documents: list[str]) -> list[list[float]]:
    # Массив эмбеддингов
    embeddings: list[list[float]] = []

    for start_index in range(0, len(documents), 32):
        document_batch = documents[start_index:start_index + 32]

        # Создаём эмбеддинги
        response = ollama.embed(
            model= "qwen3-embedding:0.6b",
            input=document_batch
        )

        batch_embeddings = response["embeddings"]

        if len(batch_embeddings) != len(document_batch):
            raise RuntimeError(
                "Ollama вернула неправильное количество эмбеддингов"
            )

        embeddings.extend(batch_embeddings)

        print(
            f"Созданы эмбеддинги: "
            f"{len(embeddings)}/{len(documents)}"
        )

    if len(embeddings) != len(documents):
        raise RuntimeError(
            "Количество эмбеддингов не совпадает "
            "с количеством документов"
        )

    return embeddings

# Функция, которая будет пересоздавать коллекцию при измненении эмбеддингов
# Upsert тут не подходит, так как возможно будут меняться параметры сплиттера
# Ну и еще это просто удобно, пусть будет
def recreate_collection(chroma_client):
    existing_collection_names = {
        collection
        if isinstance(collection, str)
        else collection.name
        for collection in chroma_client.list_collections()
    }

    if "course_lessons" in existing_collection_names:
        chroma_client.delete_collection(
            name="course_lessons"
        )

        print(
            f"Старая коллекция удалена"
        )

    return chroma_client.create_collection(
        name="course_lessons",
        configuration={
            "hnsw": {
                # Используем ту же метрику расстояния, что и в эмбеддере
                "space": "cosine",
            }
        },
    )


def index_course() -> None:

    text_files = get_text_files(DATA_DIRECTORY)

    ids, documents, metadatas = prepare_documents(
        text_files
    )

    # Сначала создаём все эмбеддинги и если эмбеддер выкенет ошибку, старая коллекция останется целой
    embeddings = create_embeddings(documents)

    chroma_client = PersistentClient(
        path=str(CHROMA_DIRECTORY)
    )

    # Пересоздаём коллекцию чтобы не оставлять в ней старые чанки
    collection = recreate_collection(chroma_client)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )

    indexed_count = collection.count()

    if indexed_count != len(documents):
        raise RuntimeError(
            "Не все документы были записаны в Chroma: "
            f"ожидалось {len(documents)}, записано {indexed_count}"
        )

    print()
    print("Индексация успешно завершена")

if __name__ == "__main__":
    index_course()


