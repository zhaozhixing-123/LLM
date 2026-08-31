FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ARG NANOCHAT_COMMIT=92d63d4e8bb4df75c3b71618f31ddde2378b2bcd

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NANOCHAT_BASE_DIR=/runpod-volume/nanochat_artifacts \
    NANOCHAT_DTYPE=bfloat16

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/karpathy/nanochat.git /opt/nanochat \
    && cd /opt/nanochat \
    && git checkout "${NANOCHAT_COMMIT}" \
    && rm -rf /opt/nanochat/.git

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir --no-deps -e /opt/nanochat \
    && rm /tmp/requirements.txt

WORKDIR /app
COPY handler.py /app/handler.py

ENTRYPOINT []
CMD ["python", "-u", "/app/handler.py"]
