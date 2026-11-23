FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /opt/chromabook
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
COPY data ./data
COPY tools ./tools
COPY app ./app
COPY migrations ./migrations
COPY alembic.ini ./alembic.ini
RUN pip install --upgrade pip && pip install .
RUN useradd --create-home --uid 10001 chromabook && mkdir -p /data && chown -R chromabook:chromabook /data /opt/chromabook
USER chromabook
ENV CHROMABOOK_DATA_DIR=/data
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers"]
