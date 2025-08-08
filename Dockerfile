# ========================
# BUILD ARGS
# ========================
# Use GPU image if BUILD_MODE=gpu, otherwise CPU base
# Example: docker build --build-arg BUILD_MODE=gpu -t my-mev-bot .
#          docker build --build-arg BUILD_MODE=cpu -t my-mev-bot .

ARG BUILD_MODE=cpu
ARG CUDA_VERSION=12.3.2
ARG UBUNTU_VERSION=22.04
ARG RUST_VERSION=1.81.0

# ========================
# BASE IMAGE SELECTION
# ========================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS gpu-base
FROM rust:${RUST_VERSION}-bullseye AS cpu-base

# ========================
# CHOOSE BASE
# ========================
FROM ${BUILD_MODE}-base AS builder

# ========================
# INSTALL SYSTEM DEPS
# ========================
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    cmake \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ========================
# INSTALL RUST (GPU build needs rustup)
# ========================
RUN if [ "${BUILD_MODE}" = "gpu" ]; then \
      curl https://sh.rustup.rs -sSf | sh -s -- -y && \
      export PATH="$HOME/.cargo/bin:$PATH"; \
    fi

# ========================
# INSTALL SOLANA CLI & ANCHOR
# ========================
RUN sh -c "$(curl -sSfL https://release.solana.com/stable/install)" && \
    cargo install --git https://github.com/coral-xyz/anchor avm --locked && \
    avm install latest && \
    avm use latest

# ========================
# SET WORKDIR
# ========================
WORKDIR /app

# ========================
# COPY BOT SOURCE
# ========================
COPY . .

# ========================
# BUILD BOT
# ========================
RUN cargo build --release

# ========================
# RUNTIME IMAGE
# ========================
FROM ${BUILD_MODE}-base AS runtime

# Install minimal runtime deps
RUN apt-get update && apt-get install -y \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/mev-bot /app/mev-bot

# ========================
# EXPOSE PORT (RunPod-style)
# ========================
# This is the port your bot listens on
EXPOSE 27112

# ========================
# ENTRYPOINT
# ========================
CMD ["./mev-bot"]
