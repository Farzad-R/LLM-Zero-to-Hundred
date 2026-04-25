# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a mono-repo of 9 independent LLM/RAG sub-projects and tutorials:

- `RAG-GPT` — document chat with ChromaDB + Gradio
- `WebGPT` — web-search-augmented chatbot with Streamlit
- `WebRAGQuery` — combines WebGPT and RAG-GPT with Chainlit
- `Hidden-Technical-Debt-Behind-AI-Agents` — LangGraph agentic chatbot with PostgreSQL + Docker
- `HUMAIN-advanced-multimodal-chatbot` — multimodal RAG (text, image, voice)
- `LLM-Fine-Tuning` — full fine-tuning pipeline with Chainlit UI
- `RAGMaster-LlamaIndex-vs-Langchain` — RAG technique comparison
- `Open-Source-RAG-GEMMA` — fully open-source RAG with Gemma + BAAI embeddings
- `tutorials/` — standalone notebooks and scripts (function calling, vectorization)

Each sub-project is fully self-contained with its own `requirements.txt`, `.env`, and — where present — its own `CLAUDE.md` with project-specific run commands and architecture details. The active conda environment is `refactor` (Python 3.11).

## Repo-Wide Conventions

**Commits** follow Conventional Commits:
- `feat:` new feature
- `fix:` bug fix
- `refactor:` code change with no behaviour change
- `chore:` tooling, deps, config

**Secrets** — all API keys and credentials live in each project's `.env` file (gitignored). Never hardcode secrets in source files.

## Branch Safety

The repo is currently being worked on in the `feat/openai-migration` feature branch.

- Never commit directly to `master`
- Never push to `master`
- Never checkout `master`
- All commits stay on the current feature branch
- The user will handle the final merge into `master` manually when ready
