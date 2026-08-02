# syntax=docker/dockerfile:1

########################################
# Stage 1: build the Rust binary
########################################
FROM rust:1.88-bookworm AS builder

WORKDIR /build

# Cache dependency compilation separately from source changes.
COPY Cargo.toml Cargo.lock* ./
RUN mkdir src && echo "fn main() {}" > src/main.rs \
    && cargo build --release || true

COPY src ./src
RUN touch src/main.rs && cargo build --release

########################################
# Stage 2: runtime (CUDA + cuDNN, matches onnxruntime's CUDA12/cuDNN9 requirement)
########################################
FROM nvidia/cuda:13.3.0-cudnn-runtime-ubuntu22.04 AS runtime

ARG ORT_VERSION=1.27.1
ARG ORT_ARCHIVE=onnxruntime-linux-x64-gpu_cuda13-${ORT_VERSION}.tgz

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# --- ONNX Runtime GPU (official Microsoft prebuilt, CUDA EP) ---------------
RUN curl -fsSL -o /tmp/ort.tgz \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE}" \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 \
    && rm /tmp/ort.tgz

ENV ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:${LD_LIBRARY_PATH}

# --- Non-root user + directory permissions ---------------------------------
# Fixed, known uid/gid so bind-mounted host directories (e.g. ./models) can be
# chowned predictably outside the container if needed.
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --shell /usr/sbin/nologin --no-create-home appuser

WORKDIR /app

# Model dir: readable by the app user, not writable (models are mounted/copied
# in, the app never needs to modify them). 750 on the dir, 640 on the file.
RUN mkdir -p /app/models \
    && chown -R appuser:appuser /app \
    && chmod 750 /app /app/models

COPY --from=builder --chown=appuser:appuser /build/target/release/realesrgan /app/realesrgan-api
RUN chmod 750 /app/realesrgan-api

# If you bake the model into the image instead of mounting it at runtime,
# uncomment this and make sure the file is readable-only by appuser:
# COPY --chown=appuser:appuser models/<MODEL_NAME> /app/models/
# RUN chmod 640 /app/models/<MODEL_NAME>

USER appuser

ENV MODEL_PATH=/app/models/<MODEL_NAME>
ENV RUST_LOG=info

EXPOSE 8080

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/realesrgan-api"]
