import subprocess
import soundfile as sf
from flask import Flask, request, jsonify, render_template, send_file, url_for
from werkzeug.utils import secure_filename
from config import load_dotenv
import os
from pyprojroot import here
import openai
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from io import BytesIO
from pydub import AudioSegment
from utils.modalities.tts_text_to_speech import convert_text_to_speech_cpu
load_dotenv()
app = Flask(__name__)

# Set up the path for uploaded files
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    # Render your HTML file
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    # Get the message from the user
    user_message = request.form['message']
    messages = [
        {"role": "system", "content": str(
            "You are a helpful chatbot")},
        {"role": "user", "content": str(user_message)}
    ]

    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo-16k",
        messages=messages,
        temperature=0,
    )
    chat_response = response["choices"][0]["message"]["content"]

    # Convert the chat response to audio
    audio_content = convert_text_to_speech_cpu(chat_response)
    audio_file = BytesIO(audio_content)
    audio_file_name = 'response_audio.wav'
    audio_file_path = os.path.join('static', audio_file_name)
    with open(audio_file_path, 'wb') as audio_file_on_disk:
        audio_file_on_disk.write(audio_file.getbuffer())

    # Return the chat response and the URL to the audio file
    return jsonify({
        'response': chat_response,
        'audio_url': url_for('static', filename=audio_file_name, _external=True)
    })


@app.route('/upload', methods=['POST'])
def handle_file_upload():
    if 'file' in request.files:
        file = request.files['file']
        # You can now save the file or process it as needed
        filename = secure_filename(file.filename)
        file.save(os.path.join(here("multimodal-bot"), filename))
        return {'status': 'success', 'message': 'File uploaded successfully'}
    else:
        return {'status': 'error', 'message': 'No file part'}, 400


# @app.route('/voice', methods=['POST'])
# def handle_voice():
#     if 'audio' in request.files:
#         audio_file = request.files['audio']
#         print(type(audio_file))
#         filename = secure_filename(audio_file.filename)
#         # Convert audio to WAV using pydub
#         temp_path = os.path.join(here("multimodal-bot/uploads"), filename)
#         audio_file.save(temp_path)
#         # Convert audio to WAV using pydub
#         audio = AudioSegment.from_file(temp_path)
        # wav_filename = os.path.splitext(filename)[0] + '.wav'
        # file_path = os.path.join('multimodal-bot', wav_filename)

        # # Save the converted audio file as WAV
        # audio.export(file_path, format='wav')

        # # try:
        # # Now try reading the saved file
        # waveform, sampling_rate = sf.read(file_path, dtype='float32')
        # # except Exception as e:
        # # print("here")
        # # os.remove(file_path)
        # # return jsonify({'status': 'error', 'message': 'Failed to read audio file', 'error': str(e)}), 500
        # os.remove(temp_path)
        # os.remove(file_path)
        # # Load the Whisper model and processor
        # processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        # model = WhisperForConditionalGeneration.from_pretrained(
        #     "openai/whisper-tiny.en")
        # # Process the waveform with the Whisper model
        # input_features = processor(
        #     waveform, sampling_rate=sampling_rate, return_tensors="pt").input_features
        # predicted_ids = model.generate(input_features)
        # transcription = processor.batch_decode(
        #     predicted_ids, skip_special_tokens=True)
        # # Clean up: remove the saved audio file if you don't need it anymore
        # os.remove("recording.wav")
        # print(transcription[0])

        # return jsonify({'status': 'success', 'transcription': transcription[0]})
    # else:
        # return jsonify({'status': 'error', 'message': 'No audio part'}), 400


@app.route('/voice', methods=['POST'])
def handle_voice():
    if 'audio' in request.files:
        audio_file = request.files['audio']
        filename = secure_filename(audio_file.filename)
        temp_path = os.path.join('uploads', filename)
        audio_file.save(temp_path)

        # Use FFmpeg to convert the audio to a proper WAV format
        converted_path = os.path.join('uploads', 'converted_' + filename)
        command = ['ffmpeg', '-i', temp_path, '-ar',
                   '16000', '-ac', '1', converted_path]
        try:
            subprocess.run(command, check=True)
            # Now you can load the converted file with your audio processing library
            # For example, using pydub:
            audio = AudioSegment.from_file(converted_path)
            # Process the audio as needed...
            return jsonify({'message': 'File processed successfully'})
        except subprocess.CalledProcessError as e:
            print(e)
            return jsonify({'message': 'Error processing file'}), 500

    return jsonify({'message': 'No audio file provided'}), 400


if __name__ == '__main__':
    app.run(debug=True)
    # "check: http://127.0.0.1:5000/"
