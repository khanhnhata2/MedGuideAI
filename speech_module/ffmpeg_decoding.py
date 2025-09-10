import os
import queue
import ffmpeg
import numpy as np
import threading
from typing import Iterator, Optional, Callable
import time

# Chỉ import sounddevice nếu môi trường hỗ trợ
USE_AUDIO = True
try:
    import sounddevice as sd
except OSError:
    print("⚠ PortAudio not found. Audio playback disabled.")
    USE_AUDIO = False
except ImportError:
    print("⚠ sounddevice not installed. Audio playback disabled.")
    USE_AUDIO = False


class AudioPlayer:
    """Encapsulates audio playback functionality with complete instance isolation."""

    def __init__(self, sample_rate: int = 24000, channels: int = 1, chunk_size: int = 1024, buffersize: int = 20):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.buffersize = buffersize
        self.dtype = 'int16'
        self.audio_queue = queue.Queue(maxsize=buffersize)
        self.process = None
        self.stream = None
        self.stop_event = threading.Event()
        self.playback_thread = None
        self.is_active = False

    def _audio_callback(self, outdata, frames, time, status):
        try:
            data = self.audio_queue.get_nowait()
        except queue.Empty:
            raise sd.CallbackAbort
        outdata[:] = np.frombuffer(data, dtype=np.int16).reshape(-1, self.channels)

    def play(self, mp3_iter: Iterator[Optional[bytes]]):
        if not USE_AUDIO:
            print("Audio playback skipped (PortAudio not available).")
            return
        if self.is_active:
            self.stop()
        self.stop_event.clear()
        self.is_active = True
        self.playback_thread = threading.Thread(target=self._playback_worker, args=(mp3_iter,), daemon=True)
        self.playback_thread.start()

    def _playback_worker(self, mp3_iter: Iterator[Optional[bytes]]):
        try:
            self.process = (
                ffmpeg
                .input('pipe:0')
                .output('pipe:1', format='s16le', acodec='pcm_s16le', ac=self.channels, ar=self.sample_rate, loglevel='quiet')
                .run_async(pipe_stdin=True, pipe_stdout=True)
            )
            def feed():
                for chunk in mp3_iter:
                    if self.stop_event.is_set():
                        break
                    if not chunk:
                        break
                    if self.process.stdin:
                        self.process.stdin.write(chunk)
                if self.process.stdin:
                    self.process.stdin.close()
            threading.Thread(target=feed, daemon=True).start()

            block_bytes = self.chunk_size * self.channels * np.dtype(self.dtype).itemsize
            for _ in range(self.buffersize):
                if self.stop_event.is_set():
                    break
                pcm = self.process.stdout.read(block_bytes)
                if not pcm:
                    break
                self.audio_queue.put_nowait(pcm)

            self.stream = sd.RawOutputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=self.channels,
                dtype=self.dtype,
                callback=self._audio_callback
            )
            with self.stream:
                timeout = self.chunk_size * self.buffersize / self.sample_rate
                while not self.stop_event.is_set():
                    pcm = self.process.stdout.read(block_bytes)
                    if not pcm:
                        break
                    self.audio_queue.put(pcm, timeout=timeout)
        except Exception as e:
            print(f"Playback error: {e}")
        finally:
            self._cleanup()

    def _cleanup(self):
        self.is_active = False
        if self.process:
            try:
                self.process.terminate()
            except:
                pass
            self.process = None
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        self.stop_event.clear()

    def stop(self):
        if not self.is_active:
            return
        self.stop_event.set()
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=0.5)
        if self.playback_thread and self.playback_thread.is_alive():
            self._cleanup()

    def is_playing(self) -> bool:
        return self.is_active


def feed_ffmpeg_and_play(mp3_iter: Iterator[Optional[bytes]], stop_event: Optional[threading.Event] = None):
    player = AudioPlayer()
    if stop_event:
        def monitor_stop_event():
            while player.is_playing():
                if stop_event.is_set():
                    player.stop()
                    break
                time.sleep(0.1)
        threading.Thread(target=monitor_stop_event, daemon=True).start()
    player.play(mp3_iter)
    while player.is_playing():
        time.sleep(0.1)
