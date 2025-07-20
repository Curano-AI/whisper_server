FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.7.1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python 3.12 and other system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        ffmpeg \
        gosu \
        wget \
        ca-certificates && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev && \
    rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12 and set it as default for both `python` and `python3`
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Add non-root user
RUN useradd -m appuser

# Install Poetry
RUN python -m pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# Install dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

# Copy application code
COPY . .

RUN chown -R appuser:appuser /app

# Copy and set up entrypoint
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
