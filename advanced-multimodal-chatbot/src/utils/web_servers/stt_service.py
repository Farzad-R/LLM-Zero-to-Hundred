from flask import Flask, request, jsonify
from transformers import pipeline
import numpy as np
from load_web_service_config import LoadWebServicesConfig

WEB_SERVICE_CFG = LoadWebServicesConfig()
print("==================================")
print("Loading openai/whisper-base.en:")
print("==================================")
transcriber = pipeline("automatic-speech-recognition",
                       model="openai/whisper-base.en")

app = Flask(__name__)


@app.route('/speech_to_text', methods=['POST'])
def speech_to_text():
    """
    Endpoint for converting speech to text.

    Expects a POST request with JSON data containing:
    - 'sr': Sampling rate of the audio
    - 'y': Raw audio data as a list of floats

    Returns a JSON response containing the transcribed text.

    Example JSON request:
    {
        "sr": 16000,
        "y": [0.1, 0.2, -0.1, ...]
    }

    Example JSON response:
    {
        "text": "The transcribed text."
    }
    """
    try:
        data = request.get_json()
        sr = data['sr']
        y = np.array(data['y'], dtype=np.float32)
        y /= np.max(np.abs(y))
        text = transcriber({"sampling_rate": sr, "raw": y})["text"]
        return jsonify({'text': text}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, port=WEB_SERVICE_CFG.whisper_service_port)
