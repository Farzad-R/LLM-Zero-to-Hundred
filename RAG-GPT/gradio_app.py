import gradio as gr
from gradio_app_utils import *
with gr.Blocks() as demo:
    ######################
    # First ROW: Image
    ######################
    # with gr.Row():
    #     gr.Image("images/header.png",
    #              width=640,
    #              height=228,
    #              show_label=False,
    #              show_download_button=False,
    #              show_share_button=False,
    #              elem_id="output_image")
    with gr.Tabs():
        with gr.TabItem("RAG-GPT"):
            ##############
            # First ROW:
            ##############
            with gr.Row() as row_one:
                with gr.Column(visible=False) as reference_bar:
                    ref_output = gr.Markdown()
                    # ref_output = gr.Textbox(
                    #     lines=22,
                    #     max_lines=22,
                    #     interactive=False,
                    #     type="text",
                    #     label="References",
                    #     show_copy_button=True
                    # )

                with gr.Column() as chatbot_output:
                    chatbot = gr.Chatbot(
                        [],
                        elem_id="chatbot",
                        bubble_full_width=False,
                        height=500,
                        avatar_images=(
                            ("images/AI_RT.png"), "images/openai_.png"),
                        # render=False
                    )
                    # **Adding like/dislike icons
                    chatbot.like(ChatBot.feedback, None, None)
            ##############
            # SECOND ROW:
            ##############
            with gr.Row():
                input_txt = gr.Textbox(
                    lines=4,
                    scale=8,
                    placeholder="Enter text and press enter, or upload a PDF file",
                    container=False,
                )

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
                temperature_bar = gr.Slider(minimum=0, maximum=1, value=0, step=0.1,
                                            label="Temperature", info="Choose between 0 and 1")
                data_type_value = gr.Dropdown(
                    label="Documents Type", choices=["Preprocessed", "Uploaded"])
                clear_button = gr.ClearButton([input_txt, chatbot])
            ##############
            # Process:
            ##############
            file_msg = upload_btn.upload(fn=GradioUploadFile.process_uploaded_files, inputs=[
                upload_btn, chatbot], outputs=[input_txt, chatbot], queue=False)

            txt_msg = input_txt.submit(fn=ChatBot.respond,
                                       inputs=[chatbot, input_txt,
                                               data_type_value, temperature_bar],
                                       outputs=[input_txt,
                                                chatbot, ref_output],
                                       queue=False).then(lambda: gr.Textbox(interactive=True),
                                                         None, [input_txt], queue=False)

            txt_msg = text_submit_btn.click(fn=ChatBot.respond,
                                            inputs=[chatbot, input_txt,
                                                    data_type_value, temperature_bar],
                                            outputs=[input_txt,
                                                     chatbot, ref_output],
                                            queue=False).then(lambda: gr.Textbox(interactive=True),
                                                              None, [input_txt], queue=False)


demo.queue()
if __name__ == "__main__":
    demo.launch()
