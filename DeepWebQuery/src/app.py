"""
    This module implements a conversational application using the Chainlit library for handling chat interactions.
    The application leverages multiple utility modules, including Apputils for managing application-related tasks,
    Memory for handling chat history storage, LLMFuntionCaller, LLMSummarizer, and LLM_RAG for interacting with
    different language models, PrepareFunctions for preparing input for GPT models, and CFG for managing configuration settings.

    The application is structured around the Chainlit library, defining functions to handle chat start and message events.
    It initializes the configuration settings using the CFG class, sets up user session data, and prepares avatars for different
    participants in the conversation.

    The conversation flow involves interacting with GPT models based on user messages. The application handles input processing,
    calls the appropriate language model (LLMFuntionCaller, LLMSummarizer, or LLM_RAG), and generates responses. It manages chat
    history, system responses, and user interactions. Additionally, it includes error handling to capture and log exceptions
    during the execution of the application.

    Note: The docstring provides an overview of the module's purpose and functionality, but detailed comments within the code
    explain specific steps and logic throughout the implementation.
"""
import chainlit as cl
from utils.app_utils import Apputils
from utils.memory import Memory
from utils.llm_function_caller import LLMFuntionCaller
from utils.llm_summarizer import LLMSummarizer
from utils.llm_rag import LLM_RAG
from utils.functions_prep import PrepareFunctions
import time
from utils.load_config import LoadConfig
import traceback

APP_CFG = LoadConfig()


@cl.on_chat_start
async def on_chat_start():
    try:
        cl.user_session.set("session_time", str(int(time.time())))
        cl.user_session.set(
            "chat_history",
            [],
        )
        cl.user_session.set(
            "rag_llm",
            False,
        )
        await cl.Avatar(
            name="DeepWebGPT",
            path="public/openai.png"
        ).send()
        await cl.Avatar(
            name="Error",
            url="public/openai.png"
        ).send()
        await cl.Avatar(
            name="User",
            path="public/logo_light.png"
        ).send()
        # check and delete the previous vector database if it exists.
        try:
            Apputils.remove_directory(APP_CFG.persist_directory)
        except:  # is being used by another process
            pass
        Apputils.create_directory("memory")
        # greeting message
        await cl.Message(f"Hello! How can I help you?").send()
    except BaseException as e:
        print(f"Caught error on on_chat_start in app.py: {e}")
        traceback.print_exc()


@cl.on_message
async def on_message(message: cl.Message):
    try:
        if message.content:  # user sends a message
            # display loader spinner while performing actions
            msg = cl.Message(content="")
            await msg.send()
            # Read recent chat history (if exists)
            chat_history_lst = Memory.read_recent_chat_history(
                file_path=APP_CFG.memory_directry.format(
                    cl.user_session.get("session_time")), num_entries=APP_CFG.num_entries)
            # Prepare input for the first model (function caller)
            input_chat_history = f"# User chat history: {chat_history_lst}"
            # check for the special character.
            # two step verification
            if message.content[:2] != "**" or not cl.user_session.get("rag_llm"):
                cl.user_session.set(
                    "rag_llm",
                    False,
                )
                messages = LLMFuntionCaller.prepare_messages(
                    APP_CFG.llm_function_caller_system_role, input_chat_history, message.content)
                print("First LLM messages:", messages, "\n")
                # Pass the input to the first model (function caller)
                llm_function_caller_full_response = LLMFuntionCaller.ask(
                    APP_CFG.llm_function_caller_gpt_model, APP_CFG.llm_function_caller_temperature, messages, APP_CFG.function_json_list)
                # If function called indeed called out a function
                if "function_call" in llm_function_caller_full_response.choices[0].message.keys():
                    print("\nCalled function:",
                          llm_function_caller_full_response.choices[0].message.function_call.name)
                    print(
                        llm_function_caller_full_response.choices[0].message, "\n")
                    # Get the pythonic response of that function
                    search_result = PrepareFunctions.execute_json_function(
                        llm_function_caller_full_response)
                    # If the called function was about a specifc url answer based on whether url loader was successful or not
                    if llm_function_caller_full_response.choices[0].message.function_call.name == "search_the_requested_url":
                        # print(llm_function_caller_full_response.choices[0].message.function_call.arguments)
                        msg = cl.Message(content="")
                        if search_result == True:
                            system_response = "Sure! The url content was processed. Please start your questions with ** if you want to chat with the url content."
                            await msg.stream_token(system_response)
                            chat_history_lst = [
                                (message.content, system_response)]
                            # Set a second argument in place for handling system manipulation.
                            cl.user_session.set(
                                "rag_llm",
                                True,
                            )
                        else:  # function_result == False
                            system_response = "Sorry, I could not process the requested url. Please ask another question."
                            await msg.stream_token(system_response)
                            chat_history_lst = [
                                (message.content, system_response)]
                    else:  # The called function was not search_the_requested_url pass the web search result to the second llm.
                        messages = LLMSummarizer.prepare_messages(
                            search_result=search_result, user_query=message.content, llm_system_role=APP_CFG.llm_summarizer_system_role)
                        print("Second LLM messages:", messages, "\n")
                        llm_summarizer_full_response = LLMSummarizer.ask(
                            APP_CFG.llm_summarizer_gpt_model, APP_CFG.llm_summarizer_temperature, messages)
                        # print the response for the user
                        llm_function_summarizer_response = llm_summarizer_full_response[
                            "choices"][0]["message"]["content"]
                        await msg.stream_token(llm_function_summarizer_response)
                        chat_history_lst = [
                            (message.content, llm_function_summarizer_response)]

                else:  # No function was called
                    llm_function_caller_response = llm_function_caller_full_response[
                        "choices"][0]["message"]["content"]
                    await msg.stream_token(llm_function_caller_response)
                    chat_history_lst = [
                        (message.content, llm_function_caller_response)]

            else:  # User message started with **
                # Implement rag with a third model. Collect the memory too.
                latest_folder = Apputils.find_latest_chroma_folder(
                    folder_path=APP_CFG.persist_directory)
                messages = LLM_RAG.prepare_messages(
                    persist_directory=latest_folder, user_query=message.content, llm_system_role=APP_CFG.llm_rag_system_role, input_chat_history=input_chat_history)
                llm_rag__full_response = LLMSummarizer.ask(
                    APP_CFG.llm_rag_gpt_model, APP_CFG.llm_rag_temperature, messages)
                llm_rag_response = llm_rag__full_response["choices"][0]["message"]["content"]
                await msg.stream_token(llm_rag_response)
                chat_history_lst = [(message.content, llm_rag_response)]

            # Update chat history or create a new csv file for each chat session
            Memory.write_chat_history_to_file(chat_history_lst=chat_history_lst, file_path=APP_CFG.memory_directry.format(
                cl.user_session.get("session_time")))
    except BaseException as e:
        print(f"Caught error on on_message in app.py: {e}")
        traceback.print_exc()
