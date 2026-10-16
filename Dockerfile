FROM python:3.12-slim
ARG WITH_SLM=0
WORKDIR /app
COPY . .
RUN if [ "$WITH_SLM" = "1" ]; then \
      apt-get update && apt-get install -y --no-install-recommends build-essential cmake && \
      pip install --no-cache-dir ".[slm]" && ntsb model pull ; \
    else pip install --no-cache-dir . ; fi
EXPOSE 8000
CMD ["uvicorn", "ntsb_api.server:app", "--host", "0.0.0.0", "--port", "8000"]
