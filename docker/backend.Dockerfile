FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python commands
RUN ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# Install poetry
RUN pip3 install poetry==1.8.4

# Set working directory
WORKDIR /app

# Copy poetry files
COPY backend/pyproject.toml backend/poetry.lock ./

# Configure poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi

# Copy application code
COPY backend .

# Create necessary directories
RUN mkdir -p storage/images storage/models storage/exports data

# Make the run script executable
COPY backend/run.py ./
RUN chmod +x run.py

# Run the application
CMD ["python3", "run.py"]