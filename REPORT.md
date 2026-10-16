# Отчёт о доработке RAG API — 15.10.2026

## Что сделано

**Настоящий RAG вместо заглушек**
- Эмбеддинги `deepvk/USER-bge-m3` (1024 измерения), пакетно, fp16 на GPU. Миграция `002` переводит колонку на `vector(1024)` и очищает корпус.
- Поиск в pgvector: 40 кандидатов, затем реранкер `qilowoq/bge-reranker-v2-m3-en-ru` оставляет 5 лучших.
- Генерация через Ollama (`OllamaAnswerProvider`) с новым RAG-промптом ([app/prompt.py](app/prompt.py)).
- Сборка под GPU (H100): `docker-compose.gpu.yml`, сервис `ollama`, `Makefile`.

**Архитектура**
- `app/providers/` — пакет: `embedding.py`, `reranker.py`, `answer.py`, `base.py`. У каждой сущности абстрактный базовый класс, который сам проверяет ответ модели.
- `app/core/config.py` — настройки из окружения с проверкой при старте; `app/core/logger.py` — единые логи loguru на русском, с временем каждого этапа.

**Аудит и исправления**
- Отказоустойчивость:
  - сбой LLM → `200` с найденными цитатами, сбой реранкера → порядок векторного поиска, оба случая видны в заголовке `X-RAG-Degraded`;
  - недоступная БД → `503`;
  - ограничены тело запроса без `Content-Length` и число одновременных генераций;
  - ошибка в вызове лога больше не роняет запрос.
- Корректность:
  - запрос без слов не уходит в поиск;
  - фраза «нет ответа» распознаётся и в кавычках;
  - `.txt` и `.md` читаются в UTF-8, UTF-16 и Windows-1251.
- Безопасность: `pypdf` 5.9.0 → 6.19.0 (85 известных уязвимостей → 0), `python-multipart` 0.0.20 → 0.0.32 (6 → 0).

**Сопровождение**
- Тесты: 43 → 56, мутанты: 50 → 69, у каждого исправления есть свой тест.
- README, `.env.example`, `.gitignore`, `requirements.txt` обновлены. `.env` теперь действительно управляет настройками compose и `make`.

---

## Запуск в контейнерах (сервер с GPU)

Нужны драйвер NVIDIA 570+, NVIDIA Container Toolkit и `make`.

```sh
cp .env.example .env      # модели и параметры — здесь
make start                # сборка, скачивание моделей, запуск
make gpu-check            # API и Ollama на GPU
docker compose exec -T rag-api pytest -q                          # 56 passed
docker compose exec -T rag-api python -m scripts.mutation_check   # 69/69
```

API: <http://localhost:8000/docs>. Остальные команды: `make` (список), `make logs`, `make restart`, `make down`.

## Локальный запуск (API без контейнера)

Нужен Python 3.12. Хранилища поднимаются в Docker, API запускается на машине.

```sh
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d postgres minio

export DATABASE_URL=postgresql://rag:rag@localhost:15432/rag
export S3_ENDPOINT_URL=http://localhost:19000 AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin
uvicorn app.main:app --port 8000
```

По умолчанию провайдеры детерминированные: модели и GPU не нужны. Чтобы включить настоящие модели (на Mac они работают через MPS), установите [Ollama](https://ollama.com), скачайте модель (`ollama pull <LLM_MODEL>`) и перед `uvicorn` добавьте:

```sh
export EMBEDDING_PROVIDER=sentence-transformers RERANKER_PROVIDER=cross-encoder \
       ANSWER_PROVIDER=ollama OLLAMA_BASE_URL=http://localhost:11434
```

Тесты локально, с теми же переменными `DATABASE_URL` и `S3_*`: `pytest -q`. Тесты сами создают базу `rag_test` и бакет `rag-documents-test`.
