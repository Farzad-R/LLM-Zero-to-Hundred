import streamlit as st
from streamlit_chat import message
import openai
import yaml
from PIL import Image
from utils.cfg import load_cfg

load_cfg()


with open("configs/app_config.yml") as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)
gpt_model = app_config["gpt_model"]
temperature = app_config["temperature"]
llm_system_role = "You are a useful chatbot."

# ===================================
# Setting page title and header
# ===================================
im = Image.open("images/AI_RT.png")

st.set_page_config(
    page_title="Simple Chatbot",
    page_icon=im,
    layout="wide"
)
st.markdown("<h1 style='text-align: center;'>Simple ChatGPT</h1>",
            unsafe_allow_html=True)

# ===================================
# Initialise session state variables
# ===================================
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
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
        user_new_prompt = "# User new question: " + user_input

        messages = [
            {"role": "system", "content": str(
                llm_system_role)},
            {"role": "user", "content": str(user_input)}
        ]

        # Generate response
        response = openai.ChatCompletion.create(
            engine=gpt_model,
            messages=messages,
            temperature=temperature,
        )
        st.session_state['past'].append(user_input)
        if "content" in response.choices[0].message.keys():
            st.session_state['generated'].append(
                response["choices"][0]["message"]["content"])
        else:
            st.session_state['generated'].append(
                "Something happened, please try again later.")


if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i],
                    is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
