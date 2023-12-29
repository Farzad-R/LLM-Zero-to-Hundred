# LLM-Zero-to-Hundred

<div align="center">
  <img src="logo/AI_RT.png" alt="CAIS" width="300" height="250">
</div>
This repository showcases various applications of LLM chatbots and provides comprehensive insights into established methodologies for training and fine-tuning Language Models.

### List of projects:
- [x] [WebGPT](#WebGPT)
- [x] [RAG-GPT](#RAG-GPT)
- [x] [WebRAGQuery](#WebRAGQuery)
- [ ] [Multimodal ChatBot](#Multimodal-ChatBot): Will be added soon.
- [ ] [LLM Full Finetuning](coming-soon): Will be added soon.
- [ ] [PEFT: Parameter-Efficient Finetuning](coming-soon): Will be added soon.
- [ ] [LLM Pretraining](coming-soon): Will be added soon.

### List of tutorials
- [x] [LLM Function Calling Tutorial](#LLM-function-calling-tutorial)
- [x] [Vectorization Tutorial](#LLM-function-calling-tutorial)

General structure of the projects:

```
Project-folder
  ├── README.md           <- The top-level README for developers using this project.
  ├── HELPER.md           <- Contains extra information that might be useful to know for executing the project.
  ├── .env                <- dotenv file for local configuration.
  ├── .here               <- Marker for project root.
  ├── configs             <- Holds yml files for project configs
  ├── data                <- Contains the sample data for the project.
  ├── src                 <- Contains the source code(s) for executing the project.
  |   └── utils           <- Contains all the necesssary project's modules. 
  └── images              <- Contains all the images used in the user interface and the README file. 
```
NOTE: This is the general structure of the projects, however there might be small changes duo to the specific needs of each project.

## Project description:
<a id="WebGPT"></a>
<h3>WebGPT:</h3>
<p>
<a style=" white-space:nowrap; " href="https://github.com/Farzad-R/LLM-Zero-to-Hundred/tree/master/WebGPT"><b>WebGPT</b></a>: is a powerful tool enabling users to pose questions that require internet searches. Leveraging GPT models:

* It identifies and executes the most relevant given Python functions in response to user queries. 
* The second GPT model generates responses by combining user queries with content retrieved from the web search engine. 
* The user-friendly interface is built using Streamlit.
* The web search supports diverse searches such as text, news, PDFs, images, videos, maps, and instant responses. 
* Overcoming knowledge-cutoff limitations, the chatbot delivers answers based on the latest internet content.

Libraries: [OpenAI](https://platform.openai.com/docs/models/overview) (It uses GPT model's function calling capability) - [duckduckgo-search](https://pypi.org/project/duckduckgo-search/) - [streamlit](https://docs.streamlit.io/)

YouTube video:
- [Link](Coming soon)

</p>

<a id="RAG-GPT"></a>
<h3>RAG-GPT:</h3>
<p>
<a style=" white-space:nowrap; " href="https://github.com/Farzad-R/LLM-Zero-to-Hundred/tree/master/RAG-GPT"><b>RAG-GPT</b></a>: is a chatbot that enables you to chat with your documents (PDFs and Doc). The chatbot offers versatile usage through three distinct methods:

1. **Offline Documents**: Engage with documents that you've pre-processed and vectorized. These documents can be seamlessly integrated into your chat sessions.
2. **Real-time Uploads:** Easily upload documents during your chat sessions, allowing the chatbot to process and respond to the content on-the-fly.
3. **Summarization Requests:** Request the chatbot to provide a comprehensive summary of an entire PDF or document in a single interaction, streamlining information retrieval.

Libraries: [OpenAI](https://platform.openai.com/docs/models/overview) - [Langchain](https://python.langchain.com/docs/get_started/quickstart) - [ChromaDB](https://www.trychroma.com/) - [Gradio](https://www.gradio.app/guides/quickstart) 

YouTube video:
- [Link](https://www.youtube.com/watch?v=1FERFfut4Uw&t=3s)
</p>

<a id="WebRAGQuery"></a>
<h3>WebRAGQuery: Combining WebGPT and RAG-GPT:</h3>
<p>
<a style=" white-space:nowrap; " href="https://github.com/Farzad-R/LLM-Zero-to-Hundred/tree/master/WebRAGQuery"><b>WebRAGQuery</b></a> is a chatbot that goes beyond typical internet searches. Built on the foundations of WebGPT and RAG-GPT, this project empowers users to delve into the depths of both general knowledge and specific URL content.

Key Features:</br>

* Intelligent Decision-Making: Our model intelligently decides whether to answer user queries based on its internal knowledge base or execute relevant Python functions.
* Dynamic Functionality: Identify and execute the most pertinent Python functions in response to user queries, expanding the scope of what the chatbot can achieve.
* Web-Integrated Responses: The second GPT model seamlessly combines user queries with content retrieved from web searches, providing rich and context-aware responses.
* Website-Specific Queries: When users inquire about a specific website, the model dynamically calls a function to load, vectorize, and create a vectordb from the site's content.
* Memory: WebRAGQuery boasts a memory feature that allows it to retain information about user interactions. This enables a more coherent and context-aware conversation by keeping track of previous questions and answers.
* Vectordb Interactions: Users can query the content of the vectordb by starting their questions with ** and exit the RAG conversation by omitting ** from the query. ** can trigger the third GPT model for RAG Q&A.
* Chainlit Interface: The user-friendly interface is built using Chainlit, enhancing the overall user experience.
* Diverse Search Capabilities: WebRAGQuery supports a variety of searches, including text, news, PDFs, images, videos, maps, and instant responses.
* Overcoming Knowledge-Cutoff Limitations: This chatbot transcends knowledge-cutoff limitations, providing answers based on the latest internet content and even allowing users to ask questions about webpage content.

YouTube video:
- [Link](Coming soon)
</p>

<a id="Multimodal-ChatBot"></a>
<h3>Multimodal-ChatBot:</h3>
This project aims to develop a versatile chatbot capable of processing text and voice inputs, generating multimodal responses (text, voice, images), and conducting web searches to deliver accurate information. Additionally, the chatbot will be equipped to analyze documents and respond to related queries.

## Tutorial description:
<a id="LLM-function-calling-tutorial"></a>
<h3>LLM Function Calling Tutorial:</h3>
<p>
<a style=" white-space:nowrap; " href="https://github.com/Farzad-R/LLM-Zero-to-Hundred/tree/master/tutorials/LLM-function-calling-tutorial"><b>LLM-function-calling-tutorial</b></a>:
    showcases the capacity of GPT models to produce executable functions in JSON format. It illustrates this capability through a practical example involving the utilization of Python with the GPT model.

Libraries: [OpenAI](https://platform.openai.com/docs/models/overview)

YouTube video:
- [Link](https://www.youtube.com/watch?v=P3bNGBTDiKM&t=3s)
</p>
<a id="LLM-function-calling-tutorial"></a>
<h2>Visualizing Text Vectorization:</h2>
<p>
<a style=" white-space:nowrap; " href="https://github.com/Farzad-R/LLM-Zero-to-Hundred/tree/master/tutorials/vectorization_tutorial"><b>Visualizing Text Vectorization</b></a>: provides a comprehensive visualization of text vectorization and demonstrates the power of vector search. It further explores the vectorization on both OpenAi `text-embedding-ada-002` and the open source `BAAI/bge-large-zh-v1.5` model.

Libraries: [OpenAI](https://platform.openai.com/docs/models/overview) - [HuggingFace](https://huggingface.co/BAAI/bge-large-zh-v1.5)

YouTube video:
- [Link](https://www.youtube.com/watch?v=sxBr_afsvb0&t=454s)
</p>

Slides:
- [Link](https://github.com/Farzad-R/LLM-Zero-to-Hundred/blob/master/presentation/slides.pdf)

## Running each project
To run the projects, you will need to install the required libraries. Follow the steps below to get started:

1. Clone the repository and navigate to the project directory.
```
git clone https://github.com/Farzad-R/LLM-Zero-to-Hundred.git
cd <yourproject>
```
2. Create a new virtual environment using a tool like virtualenv or conda, and activate the environment:
```
conda create --name projectenv python=3.11
conda activate projectenv
```
3. Install the required libraries using the following commands:
```
pip install -r requirements.txt
```
4. Then
```
cd <to each directory>
```
Follow the instructions provided for that specific project.

