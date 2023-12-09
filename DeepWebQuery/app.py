import chainlit as cl
from chainlit.input_widget import Select
from utils.app_utils import Apputils
from utils.memory import Memory
from utils.llm_function_caller import LLMFuntionCaller
from utils.llm_summarizer import LLMSummarizer
from utils.llm_rag import LLM_RAG
from utils.functions_prep import PrepareFunctions
import time
from config_loader import CFG
APP_CFG = CFG()


@cl.on_chat_start
async def on_chat_start():
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
    Apputils.remove_directory(APP_CFG.persist_directory)
    # greeting message
    await cl.Message(f"Hello! How can I help you?").send()


@cl.on_message
async def on_message(message: cl.Message):
    if message.content:  # user sends a message
        # display loader spinner while performing actions
        msg = cl.Message(content="")
        await msg.send()
        # check for the special character.
        # Read recent chat history (if exists)
        chat_history_lst = Memory.read_recent_chat_history(
            file_path=APP_CFG.memory_directry.format(
                cl.user_session.get("session_time")), num_entries=APP_CFG.num_entries)
        # Prepare input for the first model (function caller)
        input_chat_history = f"# User chat history: {chat_history_lst}"
        print("Chat history:", input_chat_history, "\n")
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
                print("Called function:",
                      llm_function_caller_full_response.choices[0].message.function_call.name)
                print(llm_function_caller_full_response.choices[0].message)
                # Get the pythonic response of that function
                search_result = PrepareFunctions.execute_json_function(
                    llm_function_caller_full_response)
                # If the called function was about a specifc url answer based on whether url loader was successful or not
                if llm_function_caller_full_response.choices[0].message.function_call.name == "search_the_requested_url":
                    msg = cl.Message(content="")
                    if search_result == True:
                        automatic_response = "Sure! The url content was processed. Please start your questions with ** if you want to chat with the url content."
                        await msg.stream_token(automatic_response)
                        chat_history_lst = [
                            (message.content, automatic_response)]
                        # Set a second argument in place for handling system manipulation.
                        cl.user_session.set(
                            "rag_llm",
                            True,
                        )
                    else:  # function_result == False
                        await msg.stream_token("Sorry, I could not process the requested url. Please ask another question.")
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
            messages = LLM_RAG.prepare_messages(
                persist_directory=APP_CFG.persist_directory, user_query=message.content, llm_system_role=APP_CFG.llm_rag_system_role, input_chat_history=input_chat_history)
            llm_rag__full_response = LLMSummarizer.ask(
                APP_CFG.llm_rag_gpt_model, APP_CFG.llm_rag_temperature, messages)
            llm_rag_response = llm_rag__full_response["choices"][0]["message"]["content"]
            await msg.stream_token(llm_rag_response)
            chat_history_lst = [(message.content, llm_rag_response)]

        # Update chat history or create a new csv file for each chat session
        Memory.write_chat_history_to_file(chat_history_lst=chat_history_lst, file_path=APP_CFG.memory_directry.format(
            cl.user_session.get("session_time")))
        print("New chat history:", chat_history_lst, "\n")
