# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Copy .env file into the container
# Only needed if you’re not using docker-compose's env_file (we'll prefer that instead)

# Set environment variables automatically
ENV PYTHONUNBUFFERED=1

# Run the chatbot
CMD ["python", "src/app.py"]
