FROM python:3.13-slim

RUN useradd -m -u 1000 user \
    && mkdir -p /home/user/app \
    && chown -R user:user /home/user/app

ENV HOME=/home/user \
    VIRTUAL_ENV=/home/user/app/.venv \
    PATH=/home/user/app/.venv/bin:/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

RUN pip install --no-cache-dir --upgrade pip uv

COPY --chown=user . .

USER user

RUN uv sync --locked --no-dev

EXPOSE 7860

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
