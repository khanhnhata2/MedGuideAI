# ffmpeg_decoding.py

import queue
import ffmpeg
import sounddevice as sd
import numpy as np
import threading
from typing import Iterator, Optional, Callable
import time

class AudioPlayer:
    """Encapsulates audio playback functionality with complete instance isolation."""
    
    def __init__(self, 
                 sample_rate: int = 24000,
                 channels: int = 1,
                 chunk_size: int = 1024,
                 buffersize: int = 20):
        """
        Initialize a new audio player instance.
        
        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
            chunk_size: Size of audio chunks to process
            buffersize: Maximum number of chunks to buffer
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.buffersize = buffersize
        self.dtype = 'int16'  # PCM format
        
        # Instance-specific audio queue
        self.audio_queue = queue.Queue(maxsize=buffersize)
        
        # State tracking
        self.process = None
        self.stream = None
        self.stop_event = threading.Event()
        self.playback_thread = None
        self.is_active = False
    
    def _audio_callback(self, outdata, frames, time, status):
        """Instance-specific audio callback function."""
        if status.output_underflow:
            print('Output underflow: increase blocksize?', flush=True)
            raise sd.CallbackAbort
        
        try:
            data = self.audio_queue.get_nowait()
        except queue.Empty:
            print('Buffer is empty: increase buffersize?', flush=True)
            raise sd.CallbackAbort
            
        # Convert raw bytes to numpy array
        outdata[:] = np.frombuffer(data, dtype=np.int16).reshape(-1, self.channels)
    
    def play(self, mp3_iter: Iterator[Optional[bytes]]):
        """
        Start playback of MP3 stream.
        
        Args:
            mp3_iter: Iterator yielding MP3 chunks
        """
        if self.is_active:
            self.stop()
        
        self.stop_event.clear()
        self.is_active = True
        
        # Start playback in a background thread
        self.playback_thread = threading.Thread(
            target=self._playback_worker,
            args=(mp3_iter,),
            daemon=True
        )
        self.playback_thread.start()
    
    def _playback_worker(self, mp3_iter: Iterator[Optional[bytes]]):
        """Worker function that handles the actual playback in a background thread."""
        try:
            # Setup ffmpeg process
            self.process = (
                ffmpeg
                .input('pipe:0')
                .output('pipe:1', 
                        format='s16le', 
                        acodec='pcm_s16le', 
                        ac=self.channels, 
                        ar=self.sample_rate, 
                        loglevel='quiet')
                .run_async(pipe_stdin=True, pipe_stdout=True)
            )
            
            # Start feeding MP3 data
            def feed():
                for chunk in mp3_iter:
                    if self.stop_event.is_set():
                        break
                    if not chunk:
                        break
                    try:
                        if self.process.stdin:
                            self.process.stdin.write(chunk)
                    except BrokenPipeError:
                        break
                try:
                    if self.process.stdin:
                        self.process.stdin.close()
                except Exception:
                    pass
            
            threading.Thread(target=feed, daemon=True).start()
            
            # Buffer initial PCM data
            block_bytes = self.chunk_size * self.channels * np.dtype(self.dtype).itemsize
            for _ in range(self.buffersize):
                if self.stop_event.is_set():
                    break
                pcm = self.process.stdout.read(block_bytes)
                if not pcm:
                    break
                try:
                    self.audio_queue.put_nowait(pcm)
                except queue.Full:
                    pass
            
            # Create and start audio stream
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
                    try:
                        self.audio_queue.put(pcm, timeout=timeout)
                    except queue.Full:
                        # If queue is full, try to make space
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.audio_queue.put_nowait(pcm)
                        except queue.Full:
                            pass
            
        except Exception as e:
            print(f"Playback error: {e}")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up all resources associated with this player instance."""
        self.is_active = False
        
        # Terminate ffmpeg process
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except:
                try:
                    self.process.kill()
                except:
                    pass
            self.process = None
        
        # Clear the audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # Reset state
        self.stop_event.clear()
    
    def stop(self):
        """Stop current playback and clean up resources."""
        if not self.is_active:
            return
        
        self.stop_event.set()
        
        # Wait briefly for playback to stop
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=0.5)
        
        # Force cleanup if thread is still running
        if self.playback_thread and self.playback_thread.is_alive():
            self._cleanup()
    
    def is_playing(self) -> bool:
        """Check if this player instance is currently playing audio."""
        return self.is_active


# Backward-compatible function for existing code
def feed_ffmpeg_and_play(mp3_iter: Iterator[Optional[bytes]], 
                         stop_event: Optional[threading.Event] = None):
    """
    Legacy function for backward compatibility.
    Creates a new AudioPlayer instance and starts playback.
    
    Args:
        mp3_iter: Iterator yielding MP3 chunks
        stop_event: Optional threading.Event to control playback
    """
    player = AudioPlayer()
    
    # If a stop_event was provided, we'll monitor it in a separate thread
    if stop_event:
        def monitor_stop_event():
            while player.is_playing():
                if stop_event.is_set():
                    player.stop()
                    break
                time.sleep(0.1)
        
        threading.Thread(target=monitor_stop_event, daemon=True).start()
    
    player.play(mp3_iter)
    
    # Wait for playback to complete (for synchronous usage)
    while player.is_playing():
        time.sleep(0.1)