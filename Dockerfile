FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install -e .

# Copy rest of project
COPY . .

CMD ["watchdog", "run-btc-scalp"]
