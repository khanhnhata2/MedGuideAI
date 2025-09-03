import streamlit as st
import speech_recognition as sr
import io

def speech_to_text(wav_audio_data: bytes) -> str:
    if wav_audio_data is not None:
        # Dùng speech_recognition để nhận diện
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(wav_audio_data) as source:
            audio = recognizer.record(source)

        try:
            return recognizer.recognize_google(audio, language="vi-VN")
        except Exception as e:
            print(str(e))
