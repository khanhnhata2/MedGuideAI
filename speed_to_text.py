import streamlit as st
from io import BytesIO
import os
import tempfile


from transformers import pipeline


# Initialize pipeline once (outside the function)
transcriber = pipeline(
    "automatic-speech-recognition",
    model="vinai/PhoWhisper-small"
)

def speed_to_text():
    transcriber = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-small")
    output = transcriber("test.wav")['text']
    print(output)


def speech_to_text(audio_bytes):
    # Save audio bytes to a temporary WAV file
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            audio_bytes = audio_bytes.read()
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

            try:
                result = transcriber(tmp_path)['text']
                return result
            finally:
                # Ensure temp file is deleted even if transcription fails
                print("Failed speech_to_text")
                os.unlink(tmp_path)

    except Exception as e:
        # Handle any errors (file IO, transcription, etc.)
        print(f"Error in speech-to-text: {str(e)}")
        return None

def speech_to_text_raw_bytes(audio_bytes):
    audio_data = audio_bytes.read()
    
    try:
        print("jump in this model")
        result = transcriber(audio_data)['text']
        return result
    except Exception as e:
        print(f"Error in speech-to-text: {str(e)}")
        return None


