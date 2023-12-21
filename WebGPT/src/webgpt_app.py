"""
     This module implements a Streamlit-based web application for a chatbot with integrated function calling and interaction with GPT models.

     The application interface is organized into a Streamlit layout, which includes:
     - Page configuration settings, such as title, icon, and layout.
     - Sidebar options for selecting the GPT model (GPT-3.5 or GPT-4), a counter, and a button to clear the conversation history.
     - Containers for displaying chat history and user input.

     The main functionality of the application includes:
     - Handling user input through a text area and a submit button.
     - Managing chat history and session state variables to store generated responses, past messages, model names, and chat history.
     - Implementing function calls to GPT models (LLM function caller and LLM chatbot) based on user input.
     - Displaying chat history and generated responses in the application.

     The application is designed to simulate a conversation with the chatbot, considering user input, system responses, and function calls.
     It leverages the Streamlit framework for creating a user-friendly and interactive web interface.

     Note: The docstring provides an overview of the module's purpose and functionality, but detailed comments within the code
     explain specific components, interactions, and logic throughout the implementation.
"""
import streamlit as st
from streamlit_chat import message
from PIL import Image
from utils.load_config import LoadConfig
from utils.app_utils import Apputils

APPCFG = LoadConfig()

function_json_list = Apputils.wrap_functions()

# ===================================
# Setting page title and header
# ===================================
im = Image.open("images/AI_RT.png")

st.set_page_config(
    page_title="WebGPT",
    page_icon=im,
    layout="wide"
)
st.markdown("<h1 style='text-align: center;'>WebGPT</h1>",
            unsafe_allow_html=True)

# ===================================
# Initialise session state variables
# ===================================
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
# ==================================
# Sidebar:
# ==================================
st.sidebar.title(
    "WebGPT: GPT agent with access to the internet")
st.sidebar.image("images/AI_RT.png", use_column_width=True)
model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
counter_placeholder = st.sidebar.empty()
clear_button = st.sidebar.button("Clear Conversation", key="clear")
# ==================================
# Reset everything (Clear button)
# ==================================
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['model_name'] = []
    st.session_state['chat_history'] = []
# ===================================
# containers:
# ===================================
response_container = st.container()  # for chat history
container = st.container()  # for text box
container.markdown("""
    <style>
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            padding: 10px;
            background-color: #f5f5f5;
            border-top: 1px solid #ddd;
        }
    </style>
""", unsafe_allow_html=True)

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=25)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        # Simplify: Only keep two chat history
        st.session_state['chat_history'] = st.session_state['chat_history'][-2:]
        chat_history = "# Chat history:\n" + \
            str([x for x in st.session_state['chat_history']])
        query = f"\n\n# User new question: {user_input}"
        messages = [
            {"role": "system", "content": str(
                APPCFG.llm_function_caller_system_role)},
            {"role": "user", "content": chat_history + query}
        ]
        print(messages)
        first_llm_response = Apputils.ask_llm_function_caller(
            APPCFG.gpt_model, APPCFG.temperature, messages, function_json_list)
        st.session_state['past'].append(user_input)
        if "function_call" in first_llm_response.choices[0].message.keys():
            print("Called function:",
                  first_llm_response.choices[0].message.function_call.name)
            web_search_result = Apputils.execute_json_function(
                first_llm_response)
            web_search_results = f"\n\n# Web search results:\n{str(web_search_result)}"
            messages = [
                {"role": "system", "content": APPCFG.llm_system_role},
                {"role": "user", "content": chat_history + web_search_results + query}
            ]
            print(messages)
            second_llm_response = Apputils.ask_llm_chatbot(
                APPCFG.gpt_model, APPCFG.temperature, messages)
            try:
                st.session_state['generated'].append(
                    second_llm_response["choices"][0]["message"]["content"])
                chat_history = (
                    f"## User question: {user_input}", f"## Response: {second_llm_response['choices'][0]['message']['content']}\n")
                st.session_state['chat_history'].append(chat_history)
            except:
                st.session_state['generated'].append(
                    "An error occured, please try again later.")
                chat_history = str(
                    (f"User question: {user_input}", f"Response: An error occured, please try again later."))
                st.session_state['chat_history'].append(chat_history)
        else:
            try:
                chat_history = str(
                    (f"User question: {user_input}", f"Response: {first_llm_response['choices'][0]['message']['content']}"))
                st.session_state['chat_history'].append(chat_history)
                st.session_state['generated'].append(
                    first_llm_response["choices"][0]["message"]["content"])

            except:
                st.session_state['generated'].append(
                    "An error occured, please try again later.")
                chat_history = str(
                    (f"User question: {user_input}", f"Response: An error occured, please try again later."))
                st.session_state['chat_history'].append(chat_history)


if st.session_state['generated']:

    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i],
                    is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
