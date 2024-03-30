from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7600]["xvector"]).unsqueeze(0)
# You can replace this embedding with your own as well.

text = "Hey there, lovely viewers! Welcome back to our channel, where creativity meets inspiration and fun. Get ready to embark on another exciting journey with us as we dive into today's content filled with laughter, learning, and unforgettable moments!"

speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})

sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])