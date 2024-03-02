import gradio as gr
from utils.upload_file import UploadFile
from utils.chatbot import ChatBot
from utils.ui_settings import UISettings
from utils.load_config import LoadConfig

APPCFG = LoadConfig()
# # Prepare the LLm and Tokenizer
# tokenizer = AutoTokenizer.from_pretrained(
#     APPCFG.llm_engine, token=APPCFG.gemma_token, device=APPCFG.device)
# model = model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="google/gemma-7b-it",
#                                                      token=APPCFG.gemma_token,
#                                                      torch_dtype=torch.float16,
#                                                      device_map=APPCFG.device
#                                                      )
# app_pipeline = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer
# )
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("RAG-GEMMA"):
            ##############
            # First ROW:
            ##############
            with gr.Row() as row_one:
                with gr.Column(visible=False) as reference_bar:
                    ref_output = gr.Markdown()

                with gr.Column() as chatbot_output:
                    chatbot = gr.Chatbot(
                        [],
                        elem_id="chatbot",
                        bubble_full_width=False,
                        height=500,
                        avatar_images=(
                            ("images/test.png"), "images/Gemma-logo.png"),
                        # render=False
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
                text_submit_btn = gr.Button(value="Submit text")
                sidebar_state = gr.State(False)
                btn_toggle_sidebar = gr.Button(
                    value="References")
                btn_toggle_sidebar.click(UISettings.toggle_sidebar, [sidebar_state], [
                    reference_bar, sidebar_state])
                upload_btn = gr.UploadButton(
                    "📁 Upload PDF or doc files", file_types=[
                        '.pdf',
                        '.doc'
                    ],
                    file_count="multiple")
                clear_button = gr.ClearButton([input_txt, chatbot])
                rag_with_dropdown = gr.Dropdown(
                    label="RAG with", choices=["Preprocessed doc", "Upload doc: Process for RAG"], value="Preprocessed doc")
            ##############
            # Fourth ROW:
            ##############
            with gr.Row() as row_four:
                temperature_bar = gr.Slider(minimum=0.1, maximum=1, value=0.1, step=0.1,
                                            label="Temperature", info="Increasing the temperature will make the model answer more creatively.")
                top_k = gr.Slider(minimum=0.0,
                                  maximum=100.0,
                                  step=1,
                                  label="top_k",
                                  value=50,
                                  info="A lower value (e.g. 10) will result in more conservative answers.")
                top_p = gr.Slider(minimum=0.0,
                                  maximum=1.0,
                                  step=0.01,
                                  label="top_p",
                                  value=0.95,
                                  info=" Works together with top-k. lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.0)")

            ##############
            # Process:
            ##############
            file_msg = upload_btn.upload(fn=UploadFile.process_uploaded_files, inputs=[
                upload_btn, chatbot, rag_with_dropdown], outputs=[input_txt, chatbot], queue=False)

            txt_msg = input_txt.submit(fn=ChatBot.respond,
                                       inputs=[chatbot, input_txt,
                                               rag_with_dropdown, temperature_bar, top_k, top_p],
                                       outputs=[input_txt,
                                                chatbot, ref_output],
                                       queue=False).then(lambda: gr.Textbox(interactive=True),
                                                         None, [input_txt], queue=False)

            txt_msg = text_submit_btn.click(fn=ChatBot.respond,
                                            inputs=[chatbot, input_txt,
                                                    rag_with_dropdown, temperature_bar, top_k, top_p],
                                            outputs=[input_txt,
                                                     chatbot, ref_output],
                                            queue=False).then(lambda: gr.Textbox(interactive=True),
                                                              None, [input_txt], queue=False)


if __name__ == "__main__":
    demo.launch()
