# LLM-playground

<div align="center">
  <img src="logo/AI_RT.png" alt="CAIS" width="300" height="250">
</div>
This repository demonstrates different use cases and well-knonw techniques for LLMs.

## Project description:
<h2>WebGPT:</h2>
<p>
<a style=" white-space:nowrap; " href="https://github.com/Farzad-R/LLM-playground/tree/master/WebGPT"><b>WebGPT</b></a>: is a powerful tool enabling users to pose questions that require internet searches. Leveraging GPT models:

* It identifies and executes the most relevant given Python functions in response to user queries. 
* The second GPT model generates responses by combining user queries with content retrieved from the web search engine. 
* The user-friendly interface is built using Streamlit
* The web search supports diverse searches such as text, news, PDFs, images, videos, maps, and instant responses. 
* Overcoming knowledge-cutoff limitations, the chatbot delivers answers based on the latest internet content.
</p>

<h2>Function calling in LLAMA and GPT models:</h2>
<p>
    - <a style=" white-space:nowrap; " href="https://github.com/Farzad-R/LLM-playground/tree/master/LLM-function-calling"><b>LLM Function calling</b></a>:
    The project showcases the capacity of Large Language Models (LLMs) to produce executable functions in JSON format. It illustrates this capability through a practical example involving the utilization of Python with the GPT model.
</p>

## Running each project
To run the projects, you will need to install the required libraries. Follow the steps below to get started:

1. Clone the repository and navigate to the project directory.
```
git clone https://github.com/Farzad-R/LLM-playground.git
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

