FROM pytorch/torchserve:latest

WORKDIR /home/model-server

COPY torchserve/model-store ./model-store
COPY torchserve/config.properties .

COPY docker_requirements.txt .
RUN pip install --no-cache-dir --index-url https://pypi.org/simple -r ./docker_requirements.txt

COPY logq ./logq
COPY pyproject.toml .
RUN pip install -e .

EXPOSE 8080
EXPOSE 8081

CMD ["torchserve", \
     "--start", \
     "--disable-token-auth", \
     "--model-store", "model-store", \
     "--models", "model=archive.mar", \
     "--ts-config", "./config.properties"]