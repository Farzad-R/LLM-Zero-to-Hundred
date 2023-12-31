"""
    This module implements a conversational application using the Chainlit library for handling chat interactions.
    The application leverages multiple utility modules, including Apputils for managing application-related tasks,
    Memory for handling chat history storage, LLMFuntionCaller, LLMSummarizer, and LLM_RAG for interacting with
    different language models, PrepareFunctions for preparing input for GPT models, and CFG for managing configuration settings.

    The application is structured around the Chainlit library, defining functions to handle chat start and message events.
    It initializes the configuration settings using the CFG class, sets up user session data, and prepares avatars for different
    participants in the conversation.

    The conversation flow involves interacting with GPT models based on user messages. The application handles input processing,
    calls the appropriate language model and generates responses. It manages chat history, system responses, and user interactions.
    Additionally, it includes error handling to capture and log exceptions during the execution of the application.

    Note: The docstring provides an overview of the module's purpose and functionality, but detailed comments within the code
    explain specific steps and logic throughout the implementation.
"""
import chainlit as cl
import time
import traceback
from utils.app_utils import Apputils
from utils.memory import Memory
from utils.llm_function_caller import LLMFuntionCaller
from utils.llm_web import LLMWeb
from utils.llm_rag import LLM_RAG
from utils.functions_prep import PrepareFunctions
from utils.load_config import LoadConfig

APP_CFG = LoadConfig()


@cl.on_chat_start
async def on_chat_start():
    try:
        cl.user_session.set("session_time", str(int(time.time())))
        cl.user_session.set(
            "chat_history",
            [],
        )
        # cl.user_session.set(
        #     "rag_llm",
        #     False,
        # )
        await cl.Avatar(
            name="WebRAGQuery",
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
        await cl.Message(f"Hello, welcome to WebRagQuery! How can I help you?").send()
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
            input_chat_history = str(chat_history_lst)
            # check for the special character.
            # two step verification
            # if message.content[:2] != "**" or not cl.user_session.get("rag_llm"):
            if message.content[:2] != "**":
                # cl.user_session.set(
                #     "rag_llm",
                #     False,
                # )
                messages = LLMFuntionCaller.prepare_messages(
                    APP_CFG.llm_function_caller_system_role, input_chat_history, message.content)
                print("First LLM messages:", messages, "\n")
                # Pass the input to the first model (function caller)
                llm_function_caller_full_response = LLMFuntionCaller.ask(
                    APP_CFG.llm_function_caller_gpt_model, APP_CFG.llm_function_caller_temperature, messages, PrepareFunctions.wrap_functions())
                # If function called indeed called out a function
                if "function_call" in llm_function_caller_full_response.choices[0].message.keys():
                    print("\nCalled function:",
                          llm_function_caller_full_response.choices[0].message.function_call.name)
                    print(
                        llm_function_caller_full_response.choices[0].message, "\n")
                    # Get the pythonic response of that function
                    func_result = PrepareFunctions.execute_json_function(
                        llm_function_caller_full_response)
                    # If the user requested to prepare a URL for Q&A (RAG)
                    if llm_function_caller_full_response.choices[0].message.function_call.name == "prepare_the_requested_url_for_q_and_a":
                        msg = cl.Message(content="")
                        if func_result == True:
                            system_response = "Sure! The url content was processed. Please start your questions with ** if you want to chat with the url content."
                            await msg.stream_token(system_response)
                            chat_history_lst = [
                                (message.content, system_response)]
                            # Set a second argument in place for handling system manipulation.
                            # cl.user_session.set(
                            #     "rag_llm",
                            #     True,
                            # )
                        else:  # function_result == False
                            system_response = "Sorry, I could not process the requested url. Please ask another question."
                            await msg.stream_token(system_response)
                            chat_history_lst = [
                                (message.content, system_response)]
                    elif llm_function_caller_full_response.choices[0].message.function_call.name == "summarize_the_webpage":
                        await msg.stream_token(func_result)
                        chat_history_lst = [
                            (message.content, func_result)]
                    else:  # The called function was not prepare_the_requested_url_for_q_and_a. Pass the web search result to the second llm.
                        messages = LLMWeb.prepare_messages(
                            search_result=func_result, user_query=message.content, llm_system_role=APP_CFG.llm_summarizer_system_role, input_chat_history=input_chat_history)
                        print("Second LLM messages:", messages, "\n")
                        llm_web_full_response = LLMWeb.ask(
                            APP_CFG.llm_summarizer_gpt_model, APP_CFG.llm_summarizer_temperature, messages)
                        # print the response for the user
                        llm_web_response = llm_web_full_response[
                            "choices"][0]["message"]["content"]
                        await msg.stream_token(llm_web_response)
                        chat_history_lst = [
                            (message.content, llm_web_response)]

                else:  # No function was called. LLM function caller is using its own knowledge
                    llm_function_caller_response = llm_function_caller_full_response[
                        "choices"][0]["message"]["content"]
                    await msg.stream_token(llm_function_caller_response)
                    chat_history_lst = [
                        (message.content, llm_function_caller_response)]

            else:  # User message started with **
                latest_folder = Apputils.find_latest_chroma_folder(
                    folder_path=APP_CFG.persist_directory)
                messages = LLM_RAG.prepare_messages(
                    persist_directory=latest_folder, user_query=message.content, llm_system_role=APP_CFG.llm_rag_system_role, input_chat_history=input_chat_history)
                llm_rag_full_response = LLM_RAG.ask(
                    APP_CFG.llm_rag_gpt_model, APP_CFG.llm_rag_temperature, messages)
                llm_rag_response = llm_rag_full_response["choices"][0]["message"]["content"]
                await msg.stream_token(llm_rag_response)
                chat_history_lst = [(message.content, llm_rag_response)]

            # Update chat history or create a new csv file for each chat session
            Memory.write_chat_history_to_file(chat_history_lst=chat_history_lst, file_path=APP_CFG.memory_directry.format(
                cl.user_session.get("session_time")))
    except BaseException as e:
        print(f"Caught error on on_message in app.py: {e}")
        traceback.print_exc()
        await cl.Message("An error occured while processing your query. Please try again later.").send()
