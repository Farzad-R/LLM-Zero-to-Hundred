
---

# Design, Build, and Deploy AI Agents

This project demonstrates how to build a **real-world, agentic chatbot** using **LangGraph**, powered by **OpenAI models**, and connected to **persistent databases** for both vector-based retrieval (RAG) and memory.

It’s designed for **developers, learners, and professionals** looking to understand how to integrate real infrastructure into LLM applications.

---

## 🚀 What This Project Covers

### ✅ 1. Agentic Chatbot with LangGraph

* Built using **LangGraph**
* Supports both **RAG** and pure chat modes
* Integrates with **OpenAI GPT models** and compatible with other models such as `Gemini 2.5`,  and `Claude 3.7 Sonnet`.
* Manages memory via LangGraph’s checkpoint system

### ✅ 2. Real Databases for Production-Style Design

* **ChromaDB** for document embeddings (vector store)
* **PostgreSQL** for persistent memory and checkpointing
* Optional: **LangSmith** integration for debugging and tracing

### ✅ 3. Microservice Architecture (Docker Compose)

* Isolated containers for:

  * Chatbot (LangGraph + Gradio)
  * Chroma (vector DB)
  * PostgreSQL (memory DB)

### ✅ 4. Automated VectorDB Creation (Inside Container)

* Runs with `chromadb.HttpClient(host="chroma", port=8000)`
* Ensures the DB is built in the same environment it serves

### ✅ 5. Visual Monitoring with Weave Scope

* Visualize container interaction and communication
* Inspect container health, logs, and live topology

### ✅ 6. Development Best Practices

* `.env` for API keys
* Custom logging (INFO, DEBUG, ERROR)
* Modular file structure
* Docker best practices for scale and deployment

---

## 📁 Project Structure

```
project-root/
├── src/
│   ├── app.py
│   ├── prepare_vectordb.py
│   └── utils/
├── data/
│   ├── pdf/
│   └── vectordb/
├── config/
│   └── config.yml
├── images/
├── .env
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🧱 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Farzad-R/LLM-Zero-to-Hundred.git
cd Design-Build-and-Deploy-AI-Agents
```

### 2️⃣ Create a Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Running Locally (Dev Setup)

### ✅ Sample `.env`

```env
OPENAI_API_KEY=your_openai_key
LANGCHAIN_API_KEY=your_langchain_key
DATABASE_URI_LOCAL=postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable
```

### 🧠 Step 1: Prepare Vector DB

```bash
python src/prepare_vectordb_local.py
```

### 🚀 Step 2: Prepare the PostgreSQL Database

```bash
docker-compose -f docker-compose-only-postgres.yml up --build
```

### 🚀 Step 3: Run the Chatbot

```bash
python src/app.py
```

### Now you have:

* Started a **PostgreSQL container** for graph memory.
* Started a **ChromaDB** locally for vector retrieval
* Started the Chatbot, at [http://localhost:7860](http://localhost:7860)
---

### Log in using one of the following credentials:

```bash
"farzad_rzt": "123",
"peter_parker": "letmein",
"admin": "securepass"
```

**Note**: To modify user credentials, edit the file at utils/user_db.py.
Please be aware that this method **is not recommended** for production. For a production-ready setup, you should use a secure external database with proper authentication and encryption mechanisms to manage user credentials safely.

## Stop the container when testing is done:
```bash
docker-compose -f docker-compose-only-postgres.yml down
```

## 🐳 Running as Microservices

![high-level](images/high-level.png)

### ✅ Update `.env` for container-based connection and add

```env
DATABASE_URI_CONTAINER=postgresql://postgres:postgres@postgres:5432/postgres
```

### 🚀 Step 1: Launch All Services

```bash
docker-compose up --build
```

Or if you want to clean up old orphan containers from the one-container (Local) setup:

```bash
docker-compose down --remove-orphans
```

```bash
docker-compose up --build -d
```


### 🔧 Step 2: Build Vector DB (Inside the Container)

```bash
docker exec -it chatbot python src/prepare_vectordb_container.py
```

### Sanity Check

#### Confirm the database was created on `chroma` container
```bash
docker exec -it chroma ls /data
```

This should show something like

```bash
146785cb-ebea-4b10-a37e-5e644d194f84  chroma.sqlite3
```

#### Confirm that the volume is mounted and storing the DB

```bash
docker exec -it langgraph-postgres-2 bash
ls /var/lib/postgresql/data
```

This should show something like

```bash
root@97ca34752563:/# ls /var/lib/postgresql/data
base          pg_dynshmem    pg_logical    pg_replslot   pg_stat      pg_tblspc    pg_wal                postgresql.conf
global        pg_hba.conf    pg_multixact  pg_serial     pg_stat_tmp  pg_twophase  pg_xact               postmaster.opts
pg_commit_ts  pg_ident.conf  pg_notify     pg_snapshots  pg_subtrans  PG_VERSION   postgresql.auto.conf  postmaster.pid
```

### Now you have:

* Started a **PostgreSQL container** for graph memory
* Started a **ChromaDB container** for vector retrieval
* Started the **Chatbot container**, exposing the Gradio interface at [http://localhost:7860](http://localhost:7860)

### Stop All Services

```bash
docker-compose stop
```

or to remove all the services 
```bash
docker-compose down
```

---

## 📊 Container Visualization with Weave Scope

![weavescope](images/weavescope.png)

### Step 1: Start the Dashboard Container (Run in Ubuntu/WSL)

```bash
docker run -d --name weave-scope \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -p 4040:4040 \
  weaveworks/scope
```

### Step 2: Launch the Probe

```bash
curl -L https://git.io/scope -o scope
chmod +x scope
./scope launch
```

### View at:

👉 [http://localhost:4040](http://localhost:4040)

> ✅ This setup only worked reliably with both steps on **WSL2 + Docker Desktop**.

### Stop Both Later:

```bash
docker stop weave-scope weavescope
docker rm weave-scope weavescope
```

---

## 🧠 Architecture Overview

| Container  | Role                           |
| ---------- | ------------------------------ |
| `chatbot`  | Gradio UI + LangGraph backend  |
| `postgres` | Stores persistent graph memory |
| `chroma`   | Stores and serves vector data  |

---

## 📋 Logging

* Logs stored in `/logs`
* Rotates daily with levels: INFO, DEBUG, ERROR

---

## 🧩 Learning Goals

* Build agentic LLM systems with LangGraph
* Connect to real vector and relational DBs
* Containerize with best practices
* Monitor AI agents in real-time
* Structure AI projects for scalability

---

## 📎 Resources

* [LangGraph Documentation](https://docs.langgraph.dev/)
* [ChromaDB](https://docs.trychroma.com/)
* [OpenAI Chat Models](https://platform.openai.com/docs/guides/gpt)
* [LangSmith (Optional)](https://smith.langchain.com/)

---

## 🙌 Built by \[Farzad Roozitalab / AI RoundTable]

If you found this helpful, consider subscribing on YouTube!
Happy building! 💡

---