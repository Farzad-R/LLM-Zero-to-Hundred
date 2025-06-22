
---

# 🐳 Docker-Powered AI Project – Beginner-Friendly Guide

Welcome! 👋
If you're new to **Docker**, don't worry — this guide will walk you through **what each part of this setup does** and how all the pieces work together.

---

## 📦 What’s in This Project?

This project runs **three components (called containers)** using Docker:

1. 🧠 **PostgreSQL** — Saves chatbot memory (like past conversations)
2. 📚 **ChromaDB** — Stores your documents as searchable vectors (for RAG)
3. 🤖 **Chatbot App** — The brain: runs the AI agent + UI (Gradio)

These are all managed together using **`docker-compose.yml`**.

---

## 🚀 What Is Docker Compose?

Docker Compose lets you **run multiple containers** with a single command. You define the setup in a `docker-compose.yml` file.

---

## 🧾 docker-compose.yml — What Does It Do?

```yaml
services:
  postgres:
    image: postgres:17
```

👆 This creates a **PostgreSQL container** based on version `17`. It's your chatbot's memory store.

```yaml
    ports:
      - "5443:5432"
```

* On your machine, port **5443** will map to the container's internal port **5432** (PostgreSQL default).
* You'll use `localhost:5443` to connect from outside Docker (e.g., DBeaver or local Python).

```yaml
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: postgres
```

* Sets up the database credentials and name.

```yaml
    volumes:
      - pgdata:/var/lib/postgresql/data
```

* Data will persist in a special Docker volume called `pgdata`, even if the container stops.

---

```yaml
  chroma:
    image: chromadb/chroma
```

👆 This starts the **ChromaDB container** — it stores your embedded document vectors (used for retrieval).

```yaml
    ports:
      - "8000:8000"
```

* Maps port **8000** so your chatbot can talk to Chroma inside or outside Docker.

```yaml
    volumes:
      - ./data/vectordb:/data
```

* This links a local folder (`./data/vectordb`) to the container's storage. Great for saving your data!

---

```yaml
  chatbot:
    build:
      context: .
      dockerfile: Dockerfile
```

👆 This builds the **chatbot container** from your local files using the provided `Dockerfile`.

```yaml
    ports:
      - "7860:7860"
```

* This exposes the chatbot on your browser at: `http://localhost:7860`

```yaml
    volumes:
      - .:/app
```

* Maps your current codebase to the `/app` folder **inside** the container. So when you edit files, changes show up in Docker immediately!

```yaml
    env_file:
      - .env
```

* Loads your secret keys (like `OPENAI_API_KEY`) from a `.env` file.

```yaml
    depends_on:
      - postgres
      - chroma
```

* Makes sure PostgreSQL and Chroma start **before** the chatbot runs.

---

```yaml
volumes:
  pgdata:
```

* This defines the volume used by PostgreSQL so that data is stored persistently.

---

## 🧾 Dockerfile — What Does It Do?

```Dockerfile
FROM python:3.11-slim
```

👆 Uses a lightweight Python 3.11 image as the base.

```Dockerfile
WORKDIR /app
```

👆 All commands will now happen inside the `/app` folder in the container.

```Dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

👆 Installs your Python dependencies.

```Dockerfile
COPY . .
```

👆 Copies your code files into the container.

```Dockerfile
ENV PYTHONUNBUFFERED=1
```

👆 Ensures logs are shown in real-time (useful for debugging).

```Dockerfile
CMD ["python", "src/app.py"]
```

👆 When the container starts, this runs your chatbot.

---

## 🧠 TL;DR — How It All Works Together

* Docker Compose sets up **three containers**: `chatbot`, `chroma`, `postgres`
* They **talk to each other** through internal Docker networking
* Data is saved persistently using **volumes**
* You get a working AI chatbot, memory system, and document search in a few commands

---

## ▶️ To Run Everything

In your terminal, run:

```bash
docker-compose up --build
```

Then open [http://localhost:7860](http://localhost:7860) to talk to your chatbot!

---