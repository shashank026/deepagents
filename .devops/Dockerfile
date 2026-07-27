FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_NO_CACHE=1
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

RUN addgroup --system app \
    && adduser --system --ingroup app app \
    && chown -R app:app /app /opt/venv
USER app

EXPOSE 8010
CMD ["uv", "run", "--no-sync", "uvicorn", "deep_agent.api:app", "--host", "0.0.0.0", "--port", "8010"]

