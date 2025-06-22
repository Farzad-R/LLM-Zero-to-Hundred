import os
from utils.rag import run_rag
from utils.chat import run_chat
from utils.load_config import LoadConfig
from gradio import Request
from utils.logging_setup import setup_logger
from typing import List, Tuple, Union

logger = setup_logger("chatbot")

CFG = LoadConfig()


class MainChatbot:
    @staticmethod
    def get_response(
        chatbot: list,
        message: str,
        app_functionality: str,
        session_id: str,
        request: Request
    ) -> Union[Tuple[str, List[Tuple[str, str]], None], Tuple[str, List[Tuple[str, str]]]]:
        """
        Processes a user's message and updates the chatbot history based on the selected application mode.

        Args:
            chatbot (List[Tuple[str, str]]): Chat history in (user, AI response) format.
            message (str): User's input message.
            app_functionality (str): Mode of operation, e.g. "RAG" or "Chat".
            session_id (str): Unique session identifier for the current chat.
            request (Request): Gradio request object, which contains user information.

        Returns:
            Union[Tuple[str, List[Tuple[str, str]], None], Tuple[str, List[Tuple[str, str]]]]:
                On success: a tuple with an empty string (for textbox reset) and the updated chatbot history.
                On failure or database unavailability: the same tuple plus a third `None` element.
                On exception: a tuple with an error message as the first item.
        """
        try:
            thread_id = thread_id = f"{request.username}_session_{session_id}"
            chat_session_config = {
                "configurable": {"thread_id": thread_id}
            }

            logger.info(f"User: {request.username}, Session: {thread_id}")

            if app_functionality == "RAG":
                if CFG.setting == "local" and not os.path.exists(CFG.stored_vectordb_dir):
                    chatbot.append(
                        (message, f"Please first create the vectorDB using the related prepare_vectordb module."))
                    logger.error("Local chroma connection failed", chatbot)
                    return "", chatbot, None

                if CFG.setting == "container":
                    try:
                        _ = CFG.stored_vectordb.get()
                    except Exception as e:
                        chatbot.append(
                            (message, "Chroma container may not be ready or vector DB not found.")
                        )
                        logger.error(
                            "Container chroma connection failed", exc_info=True)
                        return "", chatbot, None

                response = run_rag(
                    message, chat_session_config)

            elif app_functionality == "Chat":
                response = run_chat(message, chat_session_config)

            chatbot.append(
                (message, response))
            return "", chatbot
        except Exception as e:
            logger.error(f"{str(e)}")
            return f"Error: {str(e)}"
