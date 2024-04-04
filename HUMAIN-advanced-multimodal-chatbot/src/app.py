import gradio as gr
from utils.chatbot import ChatBot
from utils.raggpt.upload_file import UploadFile
from utils.ui_settings import UISettings

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Advanced Multimodal ChatBot"):
            ###############
            # Main App row:
            ###############
            with gr.Row() as app_row:
                with gr.Column(scale=1) as left_column:
                    app_functionality = gr.Dropdown(
                        label="Chatbot functionality",
                        choices=["GPT AI assistant",
                                 "LLAVA AI assistant (Understands images)",
                                 "RAG-GPT: RAG with processed documents",
                                 "RAG-GPT: RAG with upload documents",
                                 "RAG-GPT: Summarize a document",
                                 "WebRAGQuery: GPT + Duckduckgo search engine + Web RAG pipeline prep + Web Summarizer",
                                 "WebRAGQuery: RAG with the requested website (GPT model)",
                                 "Generate image (stable-diffusion-xl-base-1.0)",
                                 ],
                        value="GPT AI assistant", interactive=True)
                    # Define sliders
                    with gr.Accordion("LLM Parameters", open=False):
                        gr.Markdown(
                            "(Will be applied whenever it is applicable)")
                        gpt_temperature = gr.Slider(minimum=0, maximum=1, value=0, step=0.1,
                                                    label="Temperature:", info="Use 0 for RAG. 1 makes model more creative.", interactive=True)
                        llava_max_output_token = gr.Slider(
                            minimum=64, maximum=1024, value=512, step=64, label="Max output tokens:", interactive=True)
                    with gr.Accordion("RAG Parameters", open=False):
                        rag_top_k_retrieval = gr.Slider(
                            minimum=0, maximum=5, value=2, step=1, interactive=True, label="Top K:", info="Number of retrieved chunks for RAG")
                        rag_search_type = gr.Dropdown(
                            choices=["Similarity search", "mmr"],
                            label="Select the search technique:",
                            info="Both methods will be applied to RecursiveCharacterSplitter",
                            value="Similarity search",
                            interactive=True
                        )
                    input_audio_block = gr.Audio(
                        sources=["microphone"],
                        label="Submit your query using voice",
                        waveform_options=gr.WaveformOptions(
                            waveform_color="#01C6FF",
                            waveform_progress_color="#0066B4",
                            skip_length=2,
                            show_controls=True,
                        ),
                    )
                    audio_submit_btn = gr.Button(value="Submit audio")
                with gr.Column(scale=8) as right_column:
                    with gr.Row() as row_one:
                        with gr.Column(visible=False) as reference_bar:
                            ref_output = gr.Markdown(
                                label="RAG Reference Section")

                        with gr.Column() as chatbot_output:
                            chatbot = gr.Chatbot(
                                [],
                                elem_id="chatbot",
                                bubble_full_width=False,
                                height=500,
                                avatar_images=(
                                    ("images/AI_RT.png"), "images/chatbot.png"),
                                # render=False
                            )
                        with gr.Column(visible=False) as full_image:
                            image_output = gr.Image()
                            # **Adding like/dislike icons
                            chatbot.like(UISettings.feedback, None, None)
                    ##############
                    # SECOND ROW:
                    ##############
                    with gr.Row():
                        input_txt = gr.MultimodalTextbox(interactive=True, lines=2, file_types=[
                                                         "image"], placeholder="Enter message or upload file...", show_label=False)
                    ##############
                    # Third ROW:
                    ##############
                    with gr.Row() as row_two:
                        upload_btn = gr.UploadButton(
                            "📁 Upload PDF or doc files for RAG", file_types=[
                                '.pdf',
                                '.doc'
                            ],
                            file_count="multiple")
                        # sidebar_state = gr.State(False)
                        btn_toggle_sidebar = gr.Button(
                            value="References")
                        reference_bar_state = gr.State(False)
                        btn_toggle_sidebar.click(fn=UISettings.toggle_sidebar, inputs=[reference_bar_state], outputs=[
                            reference_bar, reference_bar_state])
                        clear_button = gr.ClearButton([input_txt, chatbot])

            #############
            # Process:
            #############
            file_msg = upload_btn.upload(fn=UploadFile.process_uploaded_files, inputs=[
                upload_btn, chatbot, app_functionality], outputs=[chatbot, input_txt], queue=False).then(lambda: gr.Textbox(interactive=True),
                                                                                                         None, [input_txt], queue=False)
            txt_msg = input_txt.submit(fn=ChatBot.respond,
                                       inputs=[chatbot, input_txt, app_functionality, gpt_temperature, llava_max_output_token, input_audio_block,
                                               rag_top_k_retrieval, rag_search_type],
                                       outputs=[chatbot, input_txt,
                                                ref_output],
                                       queue=False).then(lambda: gr.Textbox(interactive=True),
                                                         None, [input_txt], queue=False)

            txt_msg = audio_submit_btn.click(fn=ChatBot.respond,
                                             inputs=[chatbot, input_txt, app_functionality, gpt_temperature, llava_max_output_token, input_audio_block,
                                                     rag_top_k_retrieval, rag_search_type],
                                             outputs=[chatbot, input_txt,
                                                      ref_output],
                                             queue=False).then(lambda: gr.Textbox(interactive=True),
                                                               None, [input_txt], queue=False)

if __name__ == "__main__":
    demo.launch()
