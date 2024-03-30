# https://huggingface.co/microsoft/speecht5_tts
import gradio as gr
import torch
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

def text_to_speech(text):
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7600]["xvector"]).unsqueeze(0)
    # You can replace this embedding with your own as well.
    speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})
    sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
    return "speech.wav"

demo = gr.Interface(
    fn=text_to_speech,
    inputs=gr.components.Textbox(label='Input text', lines=5),
    outputs="audio"
)
if __name__ == '__main__':
    demo.launch()