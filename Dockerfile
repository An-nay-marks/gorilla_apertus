# 1. Use a base image with CUDA 12.8
FROM nvcr.io/nvidia/pytorch:25.03-py3

# 2. (Optional) Update apt and install pip/venv
RUN apt-get update && apt-get install -y python3-pip python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. CRITICAL: Unset the pip constraint environment variable
ENV PIP_CONSTRAINT=""

# 4. NOW upgrade pip and install your required setuptools
RUN pip install --upgrade pip setuptools==69.5.1

# Copy project into image and install packages from gorilla
WORKDIR /opt/gorilla_apertus
COPY berkeley-function-call-leaderboard/pyproject.toml berkeley-function-call-leaderboard/pyproject.toml
COPY berkeley-function-call-leaderboard/bfcl_eval berkeley-function-call-leaderboard/bfcl_eval

# Install the bfcl package in editable mode from its project root
WORKDIR /opt/gorilla_apertus/berkeley-function-call-leaderboard
RUN pip install -e .
RUN pip install -e .[oss_eval_vllm]


# 6. Create and set the work directory
RUN mkdir -p /workspace
WORKDIR /workspace
