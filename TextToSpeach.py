from queue import Queue
from threading import Thread, Lock
from numpy import abs as np_abs
from numpy import mean as np_mean
import time
from sounddevice import play as sd_play
from sounddevice import stop as sd_stop
from sounddevice import wait as sd_wait
from GUI import interface
from DataProcessing import UserData
import edge_tts
import io
import numpy as np
from librosa import load  
import Live2D
import asyncio

class AudioPlayer:
    speed: float = 0.85
    device: str = 'auto'
    emotion: str = None
    continue_playing: bool = True

    def __init__(self):
        # Remove the asyncio.run call - it's blocking and problematic
        self.audio_queue: Queue = Queue()
        self.streaming_queue: Queue = Queue()  
        self.streaming_thread: Thread = Thread(target=self._play_streaming_audio, daemon=False)
        self.streaming_thread.start()
        self.sentences_said: list[str] = []
        self.current_playback_active = False
        self.stop_current_flag = False
        self.current_response_id = 0  
        self.chunk_size = 1024
        self._lock = Lock()
        
        # Run the check in a separate thread
        Thread(target=self._run_startup_check, daemon=True).start()

    def _run_startup_check(self):
        """Run the startup check in a separate thread"""
        asyncio.run(self.initialize_edge_tts())

    async def initialize_edge_tts(self):
        audio_data = bytearray()
        a = edge_tts.Communicate(text="starting", voice=UserData.voice if UserData.voice != '' else "en-IE-EmilyNeural") 
        async for message in a.stream():
                
            if message["type"] == "audio":
                audio_data.extend(message["data"])

        float_samples, _ = load(io.BytesIO(audio_data), sr=24000, mono=True)
            
            
    def stop_current(self):
        print("Stopping current playback - setting flags and clearing queues")
        with self._lock:
            self.stop_current_flag = True
            self.current_playback_active = False
        
        # Stop any currently playing audio
        try:
            sd_stop()
        except Exception as e:
            print(f"Error stopping audio device: {e}")
        
        # Clear audio queues to prevent delayed playback
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Exception as e:
                print(f"Error clearing audio queue: {e}")
                break
                
        while not self.streaming_queue.empty():
            try:
                self.streaming_queue.get_nowait()
            except Exception as e:
                print(f"Error clearing streaming queue: {e}")
                break
        
        print("Audio queues cleared and playback stopped")
            
    def _play_streaming_audio(self):
        while self.continue_playing:
            try:
                # Check if current playback should be interrupted
                with self._lock:
                    if self.current_playback_active and self.stop_current_flag:
                        print("Interrupting current playback...")
                        sd_stop()
                        self.current_playback_active = False
                        self.stop_current_flag = False
                        continue
                    
                    if self.current_playback_active:
                        time.sleep(0.1)
                        continue
                    
                chunk_data = self.streaming_queue.get(timeout=0.1)
                if chunk_data is None:
                    break
                    
                audio_stream, text, emotion, response_id, chunk_id = chunk_data
                print(text, response_id, self.current_response_id)
                
                # Check if this response is still valid
                if response_id != self.current_response_id:
                    print(f"Skipping outdated response: {response_id} != {self.current_response_id}")
                    continue
                
                # Check for stop flag before starting playback
                with self._lock:
                    if self.stop_current_flag:
                        print("Stop flag set, skipping playback")
                        self.stop_current_flag = False
                        continue
                
                    self.current_playback_active = True
                
                try:
                    def send_volume_data() -> None:
                        # This works because the audio data is now int16 again
                        for i in range(0, len(audio_stream), self.chunk_size):
                            chunk = audio_stream[i:i + self.chunk_size]
                            volume = np_mean(np_abs(chunk))
                            normalized_volume = min(volume / 5000, 1.0)  
                            if normalized_volume > 0 and normalized_volume < 1.0:
                                Live2D.setVolume(normalized_volume)
                            # The sample rate is 24000 Hz
                            time.sleep(self.chunk_size / 24000.0)  

                    volume_thread = Thread(target=send_volume_data)
                    volume_thread.start()
                    # Play at 24000 Hz, which is edge-tts's default
                    sd_play(audio_stream, 24000)
                    
                    if emotion != '':
                        Live2D.setExpression(emotion) if emotion in UserData.expressions else Live2D.setMotiona(UserData.motions[emotion])
                        emotion = "*" + emotion + "*"
                    
                    self.sentences_said.append(text + emotion)
                    interface.gui.main_window.chat_window.add_message_spoken(text, False)
                        
                except Exception as e:
                    print(f"Error playing streaming audio: {e}")
                    
                sd_wait()  
                with self._lock:
                    self.current_playback_active = False
                
            except Exception as e:
                continue
            
    async def play_streaming(self, text: str, emotion: str, response_id: int, chunk_id: str = 'final') -> None:
        """Play audio with streaming support"""
        print(f"Starting TTS generation for response ID: {response_id}", time.time())
        
        # Check if response is still valid before starting
        if response_id != self.current_response_id:
            print("Response ID mismatch, skipping TTS generation.")
            return
            
        if self.stop_current_flag:
            print("Stopping current playback before starting new TTS generation.")
            return
            
        try:
            audio_data = bytearray()
            communicate = edge_tts.Communicate(text=text, voice=UserData.voice if UserData.voice != '' else "en-IE-EmilyNeural")
            print(f"Fetching audio data for response ID: {response_id}", time.time())
            
            async for message in communicate.stream():
                # Check for interruption more frequently during TTS generation
                if self.stop_current_flag or response_id != self.current_response_id:
                    print(f"TTS generation interrupted for response {response_id}")
                    return
                    
                if message["type"] == "audio":
                    audio_data.extend(message["data"])

            # Final check before queuing audio
            if self.stop_current_flag or response_id != self.current_response_id:
                print(f"TTS generation cancelled before queuing for response {response_id}")
                return

            
            float_samples, _ = load(io.BytesIO(audio_data), sr=24000, mono=True)
            # Convert to int16 to match the original data type and keep volume logic consistent
            samples = (float_samples * 32767).astype(np.int16)
            
            print(f"Audio data fetched for response ID: {response_id}", time.time())
            
            # Final validation before queuing
            if response_id == self.current_response_id and not self.stop_current_flag:
                self.streaming_queue.put((samples, text, emotion, response_id, chunk_id))
            else:
                print(f"Audio discarded due to interruption or ID mismatch: {response_id}")
            
        except Exception as e:
            print(f"Error in streaming TTS: {e}")


    async def play(self, text: str, emotion: str) -> None:

        audio_data = bytearray()
        communicate = edge_tts.Communicate(text=text, voice=UserData.voice if UserData.voice != '' else "en-IE-EmilyNeural")

        async for message in communicate.stream():
            if message["type"] == "audio":
                audio_data.extend(message["data"])

        # --- DELETED ---
        # audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        # samples = np.array(audio_segment.get_array_of_samples())

        # +++ ADDED +++
        # Use librosa to decode the mp3 data from memory.
        float_samples, _ = load(io.BytesIO(audio_data), sr=24000, mono=True)
        # Convert to int16 to match the original data type.
        samples = (float_samples * 32767).astype(np.int16)

        self.audio_queue.put((samples, text, emotion))

    def stop(self) -> None:
        print("Stopping audio player")
        self.continue_playing = False
        self.audio_queue.put((None, None, None))
        sd_stop()
        self.streaming_thread.join()


audio_player = AudioPlayer()