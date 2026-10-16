FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1


WORKDIR /workspace

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 rag \
    && mkdir -p /home/rag/.cache/huggingface \
    && chown -R rag:rag /home/rag/.cache

COPY app ./app
COPY migrations ./migrations
COPY tests ./tests
COPY scripts ./scripts
COPY pytest.ini ./pytest.ini

USER rag
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
