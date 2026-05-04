# ============================================================
# PyBatGym Docker Image
# Stage 1: Build BatSim C++ via Nix
# Stage 2: Python runtime with pre-built binary
# ============================================================

# ── Stage 1: Nix builder ────────────────────────────────────
FROM nixos/nix AS batsim-builder

# Configure Nix to allow flakes
RUN echo "experimental-features = nix-command flakes" >> /etc/nix/nix.conf

# Clone BatSim source (use nix shell to avoid nix-env profile conflicts with git-minimal)
RUN nix shell nixpkgs#git -c git clone https://framagit.org/batsim/batsim.git /batsim_src

# Build BatSim (takes 10-30 min, cached in layers)
WORKDIR /batsim_src
RUN nix build .#batsim

# Export binary to a fixed location
RUN cp -rL /batsim_src/result /batsim_build

# ── Stage 2: Python runtime ─────────────────────────────────
FROM ubuntu:22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && apt-get install -y \
    python3 python3-pip python3-venv \
    libboost-all-dev libzmq3-dev libczmq-dev \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Copy BatSim binary from builder stage
COPY --from=batsim-builder /batsim_build /opt/batsim
ENV PATH=/opt/batsim/bin:$PATH

# Verify BatSim
RUN batsim --version

# Create Python venv
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV VIRTUAL_ENV=/opt/venv

# Install Python dependencies (cached layer — runs BEFORE copying code)
WORKDIR /workspace
COPY pyproject.toml ./
COPY pybatgym/__init__.py ./pybatgym/__init__.py

RUN pip install --upgrade pip setuptools wheel && \
    pip install -e . && \
    pip install stable-baselines3 sb3-contrib tensorboard pytest pybatsim

# Copy full project (changes more often → last layer)
COPY . .
RUN pip install -e . --no-build-isolation

# Startup environment check
RUN python3 -c "import pybatgym; print('✅ pybatgym OK')" && \
    python3 -c "import stable_baselines3; print('✅ SB3 OK')" && \
    batsim --version && echo "✅ BatSim OK"

WORKDIR /workspace

# Default: open bash (override with docker-compose command)
CMD ["bash"]
