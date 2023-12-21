import pydub
from librosa.display import specshow
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from pydub import AudioSegment
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from scipy.io import wavfile
import wave
from pyprojroot import here
import os
from flask import Flask, request, send_file

with wave.open('arrow_x.wav', 'rb') as wav_file:
    samplerate = wav_file.getframerate()
    nframes = wav_file.getnframes()
    data = wav_file.readframes(nframes)

import librosa
# Load the WAV file
data, samplerate = librosa.load('arrow_x.wav', sr=None)

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
model.config.forced_decoder_ids = None

file_path = 'recording.wav'  # Replace with your .wav file path
audio_array, sampling_rate = librosa.load(file_path, sr=None)

input_features = processor(
    sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features

# generate token ids
predicted_ids = model.generate(input_features)
# decode token ids to text
transcription = processor.batch_decode(
    predicted_ids, skip_special_tokens=False)
['<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.<|endoftext|>']

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
[' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.']


def download_file(filename):
    # Make sure to validate the filename to prevent path traversal attacks
    # and ensure the file exists
    file_path = os.path.join(here("multimodal-bot"), filename)
    if os.path.isfile(file_path):
        return send_file(file_path, as_attachment=True)
    return {'message': 'File not found'}, 404


download_file("recording.wav")

samplerate, data = wavfile.read('arrow_x.wav')


# Load the MP3 file
audio = AudioSegment.from_file("arrow_x.wav", format="wav")


pydub.AudioSegment.converter = os.getcwd() + "\\ffmpeg.exe"
pydub.AudioSegment.ffprobe = os.getcwd() + "\\ffprobe.exe"
sound = pydub.AudioSegment.from_file(os.getcwd()+"\\recording.wav")
audio_path = ''
lib_data, lib_sample_rate = librosa.load(audio_path)
