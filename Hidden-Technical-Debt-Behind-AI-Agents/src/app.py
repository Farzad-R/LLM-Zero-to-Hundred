# https://www.gradio.app/guides/sharing-your-app#authentication

import uuid
import gradio as gr
from utils.ui_settings import UISettings
from utils.main_chatbot import MainChatbot
from utils.user_db import get_user_credentials


def authenticate(username: str, password: str) -> bool:
    user_db = get_user_credentials()
    return user_db.get(username) == password


with gr.Blocks(css=".tall-button { height: 85px; font-size: 16px; }") as demo:

    session_id = gr.State(str(uuid.uuid4()))
    session_display = gr.Markdown()  # Placeholder for session ID text

    with gr.Tabs():
        with gr.TabItem("End-to-End Agentic Chatbot"):
            demo.load(
                lambda s: f"🔒 Session ID: `{s}`", inputs=session_id, outputs=session_display)
            ##############
            # First ROW:
            ##############
            with gr.Row() as row_one:
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    height=500,
                    avatar_images=(
                        ("images/AI_RT.png"), "images/openai.png")
                )
                # **Adding like/dislike icons
                chatbot.like(UISettings.feedback, None, None)
            ##############
            # SECOND ROW:
            ##############
            with gr.Row():
                input_txt = gr.Textbox(
                    lines=4,
                    scale=8,
                    placeholder="Enter text and press enter, or upload PDF files",
                    container=False,
                )
            ##############
            # Third ROW:
            ##############
            with gr.Row() as row_two:
                text_submit_btn = gr.Button(
                    value="Submit text", elem_classes="tall-button")
                app_functionality = gr.Dropdown(
                    label="App functionality", choices=["RAG", "Chat"], value="RAG")
                clear_button = gr.ClearButton(
                    [input_txt, chatbot], elem_classes="tall-button")
            ##############
            # Process:
            ##############
            txt_msg = input_txt.submit(fn=MainChatbot.get_response,
                                       inputs=[chatbot, input_txt,
                                               app_functionality, session_id],
                                       outputs=[input_txt,
                                                chatbot],
                                       queue=False).then(lambda: gr.Textbox(interactive=True),
                                                         None, [input_txt], queue=False)

            txt_msg = text_submit_btn.click(fn=MainChatbot.get_response,
                                            inputs=[chatbot, input_txt,
                                                    app_functionality, session_id],
                                            outputs=[input_txt,
                                                     chatbot],
                                            queue=False).then(lambda: gr.Textbox(interactive=True),
                                                              None, [input_txt], queue=False)


if __name__ == "__main__":
    print("Starting Gradio app...")
    # demo.launch()
    # demo.launch(auth=[("farzad_rzt", "123"), ("peter_parker", "letmein")])
    # demo.launch(auth=authenticate)
    # demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    demo.launch(auth=authenticate, server_name="0.0.0.0",
                server_port=7860, share=False)
