FROM python:3.11-slim

WORKDIR /app

COPY docker_requirements.txt .

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
RUN pip install --no-cache-dir --index-url https://pypi.org/simple -r /app/docker_requirements.txt

COPY models ./models
COPY src ./src

COPY ml1m/dataset_stats.json ./ml1m/dataset_stats.json
COPY ml1m/item_cnt.pkl ./ml1m/item_cnt.pkl

COPY logq /app/logq
COPY pyproject.toml /app/pyproject.toml
RUN pip install -e .

ENTRYPOINT ["python", "-m", "src.predict"]