import streamlit as st
from streamlit_chat import message
import yaml
from PIL import Image
from utils.cfg import load_cfg
from utils.app_utils import Apputils

load_cfg()


with open("configs/app_config.yml") as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)
gpt_model = app_config["gpt_model"]
temperature = app_config["temperature"]
function_json_list = Apputils.wrap_functions()
llm_function_caller_system_role = app_config["llm_function_caller_system_role"]
llm_system_role = app_config["llm_system_role"]
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
st.sidebar.title("ChatBot with Function Calling")
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
        st.session_state['chat_history'].append({
            "role": "user",
            "content": user_input
        })
        # Simplify: Only keep one chat history
        if len(st.session_state['chat_history']) > 3:
            st.session_state['chat_history'] = st.session_state['chat_history'][-3:]

        # Exclude the current user's question from the chat_history list
        chat_history = "# User chat history: " + \
            str([x for x in st.session_state['chat_history'][:-1]]) + "\n\n"

        messages = [
            {"role": "system", "content": str(
                llm_function_caller_system_role + chat_history)},
            {"role": "user", "content": str(user_input)}
        ]
        first_llm_response = Apputils.ask_llm_function_caller(
            gpt_model, temperature, messages, function_json_list)
        st.session_state['past'].append(user_input)
        if "function_call" in first_llm_response.choices[0].message.keys():
            print("Called function:",
                  first_llm_response.choices[0].message.function_call.name)
            web_search_result = Apputils.execute_json_function(
                first_llm_response)
            query = user_input + "\n\n" + "# Web search results:\n\n" + \
                str(web_search_result)
            messages = [
                {"role": "system", "content": llm_system_role},
                {"role": "user", "content": query}
            ]

            print(messages)
            second_llm_response = Apputils.ask_llm_chatbot(
                gpt_model, temperature, messages)
            try:
                st.session_state['generated'].append(
                    second_llm_response["choices"][0]["message"]["content"])
                st.session_state['chat_history'].append(
                    {"role": "system", "content":
                     second_llm_response["choices"][0]["message"]["content"]}
                )
            except:
                st.session_state['generated'].append(
                    "Something happened. Please try again later.")
                st.session_state['chat_history'].append(
                    {"role": "system",
                        "content": "Something happened, please try again later."}
                )
        else:
            try:
                st.session_state['generated'].append(
                    first_llm_response["choices"][0]["message"]["content"])
                st.session_state['chat_history'].append(
                    {"role": "system", "content":
                     first_llm_response["choices"][0]["message"]["content"]}
                )

            except:
                st.session_state['generated'].append(
                    "Something happened. Please try again later.")
                st.session_state['chat_history'].append(
                    {"role": "system",
                        "content": "Something happened, please try again later."}
                )


if st.session_state['generated']:

    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i],
                    is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
