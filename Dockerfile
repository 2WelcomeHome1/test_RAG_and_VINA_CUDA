FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

WORKDIR /workspace

COPY requirements.txt constraints.txt ./
# constraints.txt pins every transitive dependency, so a rebuild installs exactly the tested set.
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt \
    && useradd --create-home --uid 10001 rag

COPY app ./app
COPY migrations ./migrations
COPY tests ./tests
COPY scripts ./scripts
COPY pytest.ini ./pytest.ini

USER rag
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
