import os
from pyprojroot import here
import traceback
import json
import chainlit as cl
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.functions_prep import PrepareFunctions
from utils.llm_function_caller import LLMFuntionCaller
from utils.memory import Memory
from utils.load_config import LoadConfig
from utils.inference import InferenceGPT

from typing import Dict
APP_CFG = LoadConfig()


model_path = 'openlm-research/open_llama_3b'
finetuned_model_dir = here(
    f"models/fine_tuned_models/CubeTriangle_open_llama_3b_2e_qa_qa")
max_input_tokens = 1000
max_length = 100

llm = AutoModelForCausalLM.from_pretrained(
    finetuned_model_dir, local_files_only=True, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)


def ask_cubetriangle_llm(query: str):
    """
    Generates a response from a Cubetriangle Company's private large language model based on the given query.

    Parameters:
    - query (str): The input query for the language model.

    Returns:
    - str: The generated response from the language model.
    """
    inputs = tokenizer(query, return_tensors="pt",
                       truncation=True, max_length=max_input_tokens).to("cuda")
    tokens = llm.generate(**inputs, max_length=max_length)
    response = tokenizer.decode(
        tokens[0], skip_special_tokens=True)[len(query):]
    return response


@cl.on_chat_start
async def on_chat_start():
    try:
        cl.user_session.set("session_time", str(int(time.time())))
        await cl.Avatar(
            name="Enterprise LLM",
            path="src/chatbot/public/logo.png"
        ).send()
        await cl.Avatar(
            name="Error",
            url="src/chatbot/public/logo.png"
        ).send()
        await cl.Avatar(
            name="User",
            path="src/chatbot/public/logo_light.png"
        ).send()
        if not os.path.exists("memory"):
            os.makedirs("memory")
        await cl.Message(f"Hello! I am the CubeTriangle ChatBot. How can I help you?").send()
    except BaseException as e:
        print(f"Caught error on on_chat_start in app.py: {e}")
        traceback.print_exc()


@cl.on_message
async def on_message(message: cl.Message):
    try:
        # display loader spinner while performing actions
        msg = cl.Message(content="")
        await msg.send()
        chat_history_lst = Memory.read_recent_chat_history(
            file_path=APP_CFG.memory_directry.format(
                cl.user_session.get("session_time")), num_entries=APP_CFG.num_entries)
        # Prepare input for the first model (function caller)
        input_chat_history = str(chat_history_lst)
        messages = LLMFuntionCaller.prepare_messages(
            APP_CFG.llm_function_caller_system_role, input_chat_history, message.content)
        print("First LLM messages:", messages, "\n")
        # Pass the input to the first model (function caller)
        llm_function_caller_full_response = LLMFuntionCaller.ask(
            APP_CFG.llm_function_caller_gpt_model, APP_CFG.llm_function_caller_temperature, messages, [PrepareFunctions.jsonschema(
                ask_cubetriangle_llm)])

        # If function called indeed called out a function
        if "function_call" in llm_function_caller_full_response.choices[0].message.keys():
            print(
                llm_function_caller_full_response.choices[0].message, "\n")
            # Get the pythonic response of that function
            func_name: str = llm_function_caller_full_response.choices[
                0].message.function_call.name
            print("\nCalled function:", func_name)
            func_args: Dict = json.loads(
                llm_function_caller_full_response.choices[0].message.function_call.arguments)
            # Call the function with the given arguments
            if func_name == 'ask_cubetriangle_llm':
                llm_response = ask_cubetriangle_llm(
                    **func_args)
            else:
                raise ValueError(f"Function '{func_name}' not found.")
            messages = InferenceGPT.prepare_messages(
                llm_response=llm_response, user_query=message.content, llm_system_role=APP_CFG.llm_inference_system_role, input_chat_history=input_chat_history)
            print("Second LLM messages:", messages, "\n")
            llm_inference_full_response = InferenceGPT.ask(
                APP_CFG.llm_inference_gpt_model, APP_CFG.llm_inference_temperature, messages)
            # print the response for the user
            llm_inference_response = llm_inference_full_response[
                "choices"][0]["message"]["content"]
            await msg.stream_token(llm_inference_response)
            chat_history_lst = [
                (message.content, llm_inference_response)]

        else:  # No function was called. LLM function caller is using its own knowledge.
            llm_function_caller_response = llm_function_caller_full_response[
                "choices"][0]["message"]["content"]
            await msg.stream_token(llm_function_caller_response)
            chat_history_lst = [
                (message.content, llm_function_caller_response)]

        # Update memory
        Memory.write_chat_history_to_file(chat_history_lst=chat_history_lst, file_path=APP_CFG.memory_directry.format(
            cl.user_session.get("session_time")))

    except BaseException as e:
        print(f"Caught error on on_message in app.py: {e}")
        traceback.print_exc()
        await cl.Message("An error occured while processing your query. Please try again later.").send()
