FROM agrigorev/zoomcamp-model:2025

RUN pip install uv

WORKDIR /code

COPY ".python-version" "pyproject.toml" "uv.lock" ./

RUN uv sync --locked

COPY "main.py" "pipeline_v1.bin" ./

EXPOSE 9696

ENTRYPOINT ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9696"]