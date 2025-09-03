
import streamlit as st
import edge_tts
import io
from concurrent.futures import ThreadPoolExecutor
import asyncio
import base64

# st.title("🎙️ Edge TTS Stream Demo (no disk save)")

# language = st.radio("Select Language", ["English", "Vietnamese"], horizontal=True)
# voice_map = {
#     "English": "en-US-JennyNeural",
#     "Vietnamese": "vi-VN-HoaiMyNeural"
# }
# voice = voice_map[language]

# text = st.text_area("Enter text to synthesize:", height=150)

def run_async_tts(input_text):
    async def generate_audio(input_text) -> bytes:
        communicate = edge_tts.Communicate(text=input_text, voice="vi-VN-HoaiMyNeural")
        audio_stream = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_stream.write(chunk["data"])
        return audio_stream.getvalue()
    
    return asyncio.run(generate_audio(input_text))

def run_audio(input_text):
    with ThreadPoolExecutor() as executor:
        print("thong jump in here")
        audio_bytes = executor.submit(run_async_tts, input_text).result()
        return audio_bytes
    
async def generate_audio(text_input):
    tts = edge_tts.Communicate(text=text_input, voice="vi-VN-HoaiMyNeural")
    audio_buffer = io.BytesIO()
    async for chunk in tts.stream():
        audio_buffer.write(chunk)
    audio_buffer.seek(0)
    return audio_buffer.read()
    

# st.audio(audio_bytes, format="audio/mp3")


# def run_audio(input_text, streamlit):
#     print("streamlit: " + streamlit)
#     audio_bytes = run_async_tts(input_text=input_text)
#     b64_audio = base64.b64encode(audio_bytes).decode()

    # audio_html = f"""
    #             <audio autoplay>
    #                 <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mpeg">
    #             </audio>
    #             """
    # streamlit.markdown(audio_html, unsafe_allow_html=True)


# if text.strip():
#     with st.spinner("Synthesizing speech..."):
#         # with ThreadPoolExecutor() as executor:
#             # audio_bytes = executor.submit(run_async_tts).result()
#         audio_bytes = run_async_tts("xin chào thế giới")
#         b64_audio = base64.b64encode(audio_bytes).decode()

#         # Inject autoplay audio using HTML
#         audio_html = f"""
#         <audio autoplay>
#             <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mpeg">
#         </audio>
#         """
#         st.markdown(audio_html, unsafe_allow_html=True)

        # st.audio(audio_bytes, format="audio/mp3")
        # st.download_button(
        #     "⬇️ Download MP3",
        #     data=audio_bytes,
        #     file_name="speech.mp3",
        #     mime="audio/mpeg"
        # )