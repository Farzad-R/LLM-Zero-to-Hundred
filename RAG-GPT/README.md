# RAG-GPT: Fostering Robust AI Conversations with retrieval Augemented generation By Integrating OpenAI GPT Model, Langchain, ChromaDB, and Gradio

**RAG-GPT** is a chatbot that enables you to chat with your documents (PDFs and Doc). The project works with two types of data:
1. With documents that you have vectorized and processed beforehand.
2. With documents that you upload while chatting with the model.

## RAG-GPT User Interface
<div align="center">
  <img src="images/RAG-GPT UI.png" alt="RAG-GPT UI">
</div>

## Project Schema
<div align="center">
  <img src="images/RAG-GPT_schema.png" alt="Schema">
</div>

* NOTE: This project is currently set up as a **demo**. As such, the document management is simplified and not suitable for production environments.

## Document Storage
Documents are stored in two separate folders within the `data` directory:
- `data/docs_2`: For files that you want to **upload**.
- `data/docs`: For files that should be **processed in advance**.

## Server Setup
The `serve.py` module leverages these folders to create an **HTTPS server** that hosts the PDF files, making them accessible for user viewing.

## Database Creation
Vector databases (vectorDBs) are generated within the `data` folder, facilitating the project's functionality.

## Important Considerations
- The current file management system is intended for **demonstration purposes only**.
- It is **strongly recommended** to design a more robust and secure document handling process for any production deployment.
- Ensure that you place your files in the correct directories (`data/docs_2` and `data/docs`) for the project to function as intended.

## Running the Project

To get the project up and running, you'll need to set up your environment and install the necessary dependencies. You can do this in two ways:

### Option 1: Using the Parent Directory Instructions

Follow the instruction on the [parent directory](https://github.com/Farzad-R/LLM-playground/tree/master) to create an environment and install required libraries. 

### Option 2: Installing Dependencies Individually
If you prefer to install the dependencies individually, run the following command:

```
pip install gradio==4.5.0 langchain==0.0.339 openai==0.28.0 chromadb==0.4.18 PyYAML pypdf==3.17.1
```

1. **Configuration and Execution**
* Open cfg.py and fill in your GPT API credentials.

2. **Activate Your Environment.**
3. **Ensure you are in the RAG-GPT directory**
4. **Run the Application:**

In Terminal 1:
```
python serve.py
```

In Terminal 2:
```
python gradio_app.py
```
5. Chat with the RAG-GPT

YouTube video:
- [Link](Coming soon)

Slides:
- [Link](https://github.com/Farzad-R/LLM-Zero-to-Hundred/blob/master/presentation/slides.pdf)

Extra read:
- [GPT model](https://platform.openai.com/docs/models/overview) 
- [Gradio](https://www.gradio.app/guides/quickstart)
- [Langchain](https://python.langchain.com/docs/get_started/quickstart)
- [ChromaDB](https://www.trychroma.com/)