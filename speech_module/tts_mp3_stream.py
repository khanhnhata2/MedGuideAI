import json
from typing import Iterator, Optional
import edge_tts
import threading

import ffmpeg
import io

_test_text = """Kỷ Nguyên AI – Điều Gì Đang Diễn Ra?

Chúng ta đang sống trong kỷ nguyên AI – thời đại mà trí tuệ nhân tạo (AI) không còn là viễn tưởng mà đã hiện diện khắp nơi: từ trợ lý ảo như Siri, ChatGPT đến xe tự lái và máy dịch ngôn ngữ.

AI là hệ thống máy tính có thể học hỏi, phân tích và đưa ra quyết định giống con người. Điều này giúp tiết kiệm thời gian, tăng hiệu suất và mở ra nhiều cơ hội trong giáo dục, y tế, kinh doanh...

Tuy nhiên, AI cũng đặt ra câu hỏi về việc làm, đạo đức và quyền riêng tư. Vì vậy, việc hiểu và ứng dụng AI một cách có trách nhiệm là rất quan trọng.

Kỷ nguyên AI không phải điều đáng sợ, mà là cơ hội để con người hợp tác với công nghệ, phát triển bền vững hơn trong tương lai."""


class TTSController:
    """Controller object to allow cancelling a running TTS stream/playback."""

    def __init__(self) -> None:
        self._stop_event = threading.Event()

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def stop(self) -> None:
        """Signal cancellation. Playback/stream generators check this event and stop."""
        self._stop_event.set()


def tts_stream(text: str, stop_event: Optional[threading.Event] = None) -> Iterator[Optional[bytes]]:
    """Yield MP3 chunks produced by edge-tts. If `stop_event` is set the generator exits early.

    This generator yields raw MP3 chunk bytes. For Streamlit, collect the bytes and call
    `st.audio(mp3_bytes, format='audio/mp3')`.
    """
    communicate = edge_tts.Communicate(text, voice="vi-VN-HoaiMyNeural")
    stop_event = stop_event or threading.Event()
    # with open("output.mp3", "wb") as fa:
    for chunk in communicate.stream_sync():
        if stop_event.is_set():
            # Early exit on cancel request
            yield None
            break
        # Prepare a dict without the 'data' field if it's audio
        # chunk_info = {k: v for k, v in chunk.items() if k != "data"}
        # with open("chunks_output.txt", "a", encoding="utf-8") as f:
        #     f.write(json.dumps(chunk_info, ensure_ascii=False) + "\n")
        if chunk.get("type") == "audio" and "data" in chunk:
            # fa.write(chunk["data"])
            yield chunk["data"]
    # signal completion
    yield None

def playback_mp3_stream(mp3_chunks: Iterator[Optional[bytes]], stop_event: Optional[threading.Event] = None):
    """Play MP3 chunks using ffplay subprocess. Honor `stop_event` to cancel playback."""
    import subprocess

    stop_event = stop_event or threading.Event()

    ffplay = subprocess.Popen(
        ['ffplay', '-nodisp', '-autoexit', '-'],
        stdin=subprocess.PIPE
    )

    # Stream chunks into ffplay, stop if stop_event is set
    for chunk in mp3_chunks:
        if stop_event.is_set():
            break
        if chunk is None:
            break
        if ffplay.stdin:
            ffplay.stdin.write(chunk)
    try:
        if ffplay.stdin:
            ffplay.stdin.close()      # Close stdin to signal end of stream
    except Exception:
        pass
    ffplay.wait()

def playback_mp3_stream_ffmpeg(mp3_chunks: Iterator[Optional[bytes]], stop_event: Optional[threading.Event] = None):
    """Feed MP3 chunks into an ffmpeg process; this example simply forwards the bytes.
    It will stop early if `stop_event` is set."""
    stop_event = stop_event or threading.Event()

    process = (
        ffmpeg
        .input('pipe:0')
        .output('pipe:1', format='mp3')
        .run_async(pipe_stdin=True, pipe_stdout=True)
    )

    for chunk in mp3_chunks:
        if stop_event.is_set():
            break
        if chunk is None:
            break
        if process.stdin:
            process.stdin.write(chunk)
    try:
        if process.stdin:
            process.stdin.close()
    except Exception:
        pass
    process.wait()

def main():
    controller = TTSController()
    try:
        playback_mp3_stream(tts_stream(_test_text, stop_event=controller.stop_event), stop_event=controller.stop_event)
    except KeyboardInterrupt:
        controller.stop()
        print("Stopped by user")
    print("Playback finished.")

if __name__ == "__main__":
    # asyncio.run(main())
    main()