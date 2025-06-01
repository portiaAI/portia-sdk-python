# Build stage with Rust compiler
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# Install Rust and required build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libssl-dev \
    pkg-config \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rm -rf /var/lib/apt/lists/*

# Add Rust to PATH
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app
COPY pyproject.toml uv.lock LICENSE README.md /app/

# Build dependencies with Rust available
RUN uv sync --no-dev

# Production stage - clean bookworm-slim image
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim


WORKDIR /app

# Copy the built virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv
COPY portia /app/portia
COPY .env /app/.env
COPY portia_gui.py /app/portia_gui.py
COPY portia_gui.tcss /app/portia_gui.tcss

# Ensure uv can find the virtual environment
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONPATH=/app

CMD ["textual", "serve", "-h", "0.0.0.0", "-p", "8000", "portia_gui.py"]
