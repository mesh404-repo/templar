# Templar Evaluation Environment
# Usage: docker build --network=host -f environments/templar/Dockerfile -t templar-eval:latest .
# Run from repo root so hparams.json is accessible

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash appuser
WORKDIR /app

COPY --chown=appuser:appuser environments/templar/requirements.txt /app/
USER appuser
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy hparams and download model/dataset based on config
COPY --chown=appuser:appuser hparams/hparams.json /app/

RUN python -c "\
import json; \
cfg = json.load(open('/app/hparams.json')); \
model = cfg['benchmark_model_name']; \
from transformers import AutoModelForCausalLM, AutoTokenizer; \
print(f'Downloading model: {model}'); \
AutoModelForCausalLM.from_pretrained(model, torch_dtype='auto', trust_remote_code=True); \
AutoTokenizer.from_pretrained(model, trust_remote_code=True); \
print('Model cached')"

COPY --chown=appuser:appuser environments/templar/cache_dataset.py /app/
RUN python -c "\
import json, os; \
cfg = json.load(open('/app/hparams.json')); \
os.environ['DATASET_NAME'] = cfg['benchmark_dataset_name']; \
os.environ['NUM_SAMPLES'] = str(cfg['benchmark_data_samples']); \
exec(open('/app/cache_dataset.py').read())" || [ -f /home/appuser/.cache/templar/dataset.json ]

COPY --chown=appuser:appuser environments/templar/env.py /app/

ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PATH="/home/appuser/.local/bin:${PATH}"
ENV CACHED_DATASET_PATH="/home/appuser/.cache/templar/dataset.json"
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "env:app", "--host", "0.0.0.0", "--port", "8000"]
