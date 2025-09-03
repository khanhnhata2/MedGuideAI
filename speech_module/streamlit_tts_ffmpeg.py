# streamlit_tts_ffmpeg.py

import streamlit as st
import threading
import time
import uuid
from ffmpeg_decoding import AudioPlayer
from tts_mp3_stream import tts_stream

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}  # Dictionary with message IDs as keys

# Fragment for playback controls - this is the key to solving the button issue!
@st.fragment
def playback_controls(message_id: str, message_content: str):
    """Isolated fragment for managing playback controls for a single message."""
    # Get or create player reference in session state
    if 'audio_players' not in st.session_state:
        st.session_state.audio_players = {}
    
    # Initialize player for this message if needed
    if message_id not in st.session_state.audio_players:
        st.session_state.audio_players[message_id] = None
    
    player = st.session_state.audio_players[message_id]
    is_playing = player is not None and player.is_playing()
    
    # Display appropriate button based on playback state
    if is_playing:
        if st.button("⏹️", key=f"stop_{message_id}"):
            # Stop this specific playback
            if player:
                player.stop()
            # Clear player reference
            st.session_state.audio_players[message_id] = None
            # st.rerun(scope="fragment")
            # The fragment will automatically rerun and show the Play button
            
        if isinstance(player.playback_thread, threading.Thread) and player.playback_thread.is_alive():
            # time.sleep(0.1)
            # st.rerun(scope="fragment")
            pass
        else:
            st.session_state.audio_players[message_id] = None
        time.sleep(0.1)
        st.rerun(scope="fragment")

    else:
        if st.button("🔊", key=f"play_{message_id}"):
            # First stop any active playback (enforce only one playback at a time)
            for msg_id, play_task in st.session_state.audio_players.items():
                p, _ = play_task or (None, None)
                if p is not None:
                    p.stop()
                    st.session_state.audio_players[msg_id] = None

            # Create new player for this message
            player = AudioPlayer()
            # Start playback in background thread
            def playback():
                player.play(tts_stream(message_content))
            
            t = threading.Thread(target=playback, daemon=True)
            st.session_state.audio_players[message_id] = player
            t.start()
            st.rerun(scope="fragment")

# Main app layout
st.title("💬 TTS Chat Assistant")

# Display chat history
for message_id, message in st.session_state.chat_history.items():
    with st.chat_message("user"):
        st.write(message["chat_content"])
        # Each message's controls run in their own isolated fragment
        playback_controls(message_id, message["chat_content"])

# Chat input at the bottom
if prompt := st.chat_input("Type a message..."):
    # Generate a unique ID for this message
    message_id = str(uuid.uuid4())
    # Add to chat history
    st.session_state.chat_history[message_id] = {
        "chat_content": prompt
    }
    st.rerun()