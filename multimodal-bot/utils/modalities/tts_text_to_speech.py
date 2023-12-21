# https://colab.research.google.com/drive/11voeyNXpvOuZ8h3r2c_eKO99M8ulyRfE#scrollTo=9ONq36Z1kcDE
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch


def convert_text_to_speech_cpu(text: str):
    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

    embeddings_dataset = load_dataset(
        "Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(
        embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    speech = synthesiser(text, forward_params={
                         "speaker_embeddings": speaker_embedding})
    sf.write("database/tts/response_speech.wav",
             speech["audio"], samplerate=speech["sampling_rate"])
    return speech["audio"]


def convert_text_to_speech_gpu(text: str):
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise SystemError("CUDA is not available on this system.")

    synthesiser = pipeline(
        "text-to-speech", "microsoft/speecht5_tts", device=0)

    embeddings_dataset = load_dataset(
        "Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(
        embeddings_dataset[7306]["xvector"]).unsqueeze(0).to('cuda')
    speech = synthesiser(text, forward_params={
                         "speaker_embeddings": speaker_embedding})
    sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
    return speech["audio"]
