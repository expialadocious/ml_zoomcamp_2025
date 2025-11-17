FROM python:3.12.4-slim

# 1) Install system dependencies (needed for xgboost compilation and performance)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2) Install uv (your dependency manager)
RUN pip install --no-cache-dir uv

# 3) Set working directory
WORKDIR /app

# 4) Copy dependency files first (for build caching)
COPY pyproject.toml uv.lock ./

# 5) Install dependencies
RUN uv sync --frozen --no-dev

# 6) Copy your app code + model file
COPY predict.py pipeline_v2.bin ./

# 7) Expose FastAPI port
EXPOSE 9696

# 8) Command to run the FastAPI app
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9696"]
