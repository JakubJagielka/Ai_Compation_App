from typing import Iterable, Optional
from typing import List, Union
from ctypes import c_bool
from scipy.signal import resample
from scipy import signal
import collections
import numpy as np
import traceback
import threading
import itertools
import platform
import pyaudio
import logging
import struct
import halo
import time
import copy
import os
import gc
import multiprocessing as mp
import webrtcvad
from soundfile import write


INIT_MODEL_TRANSCRIPTION = "tiny"
INIT_MODEL_TRANSCRIPTION_REALTIME = "tiny"
INIT_REALTIME_PROCESSING_PAUSE = 0.2
INIT_SILERO_SENSITIVITY = 0.4
INIT_WEBRTC_SENSITIVITY = 3
INIT_POST_SPEECH_SILENCE_DURATION = 0.6
INIT_MIN_LENGTH_OF_RECORDING = 0.5
INIT_MIN_GAP_BETWEEN_RECORDINGS = 0
INIT_WAKE_WORDS_SENSITIVITY = 0.6
INIT_PRE_RECORDING_BUFFER_DURATION = 1.0
INIT_WAKE_WORD_ACTIVATION_DELAY = 0.0
INIT_WAKE_WORD_TIMEOUT = 5.0
INIT_WAKE_WORD_BUFFER_DURATION = 0.1
ALLOWED_LATENCY_LIMIT = 10
vad = webrtcvad.Vad(3)
TIME_SLEEP = 0.02
SAMPLE_RATE = 16000
BUFFER_SIZE = 512
INT16_MAX_ABS_VALUE = 32768.0
INIT_HANDLE_BUFFER_OVERFLOW = False
if platform.system() != 'Darwin':
    INIT_HANDLE_BUFFER_OVERFLOW = True


class AudioToTextRecorder:
    def __init__(self,
                 model: str = INIT_MODEL_TRANSCRIPTION,
                 language: str = "",
                 compute_type: str = "default",
                 input_device_index: int = None,
                 gpu_device_index: Union[int, List[int]] = 0,
                 device: str = "cuda",
                 on_recording_start=None,
                 on_recording_stop=None,
                 on_transcription_start=None,
                 ensure_sentence_starting_uppercase=True,
                 ensure_sentence_ends_with_period=True,
                 use_microphone=True,
                 spinner=True,
                 level=logging.WARNING,
                 apikey = None,

                 # Realtime transcription parameters
                 enable_realtime_transcription=False,
                 realtime_model_type=INIT_MODEL_TRANSCRIPTION_REALTIME,
                 realtime_processing_pause=INIT_REALTIME_PROCESSING_PAUSE,
                 on_realtime_transcription_update=None,
                 on_realtime_transcription_stabilized=None,

                 # --- NEW: Parameters for adjustable voice detection ---
                 voice_activity_threshold: float = 4000.0, # User-defined amplitude threshold
                 on_audio_level_update=None, # Callback for UI updates

                 # Voice activation parameters
                 silero_sensitivity: float = INIT_SILERO_SENSITIVITY,
                 silero_use_onnx: bool = False,
                 webrtc_sensitivity: int = INIT_WEBRTC_SENSITIVITY,
                 post_speech_silence_duration: float = (
                         INIT_POST_SPEECH_SILENCE_DURATION
                 ),
                 min_length_of_recording: float = (
                         INIT_MIN_LENGTH_OF_RECORDING
                 ),
                 min_gap_between_recordings: float = (
                         INIT_MIN_GAP_BETWEEN_RECORDINGS
                 ),
                 pre_recording_buffer_duration: float = (
                         INIT_PRE_RECORDING_BUFFER_DURATION
                 ),
                 on_vad_detect_start=None,
                 on_vad_detect_stop=None,

                 # Wake word parameters
                 wakeword_backend: str = "pvporcupine",
                 openwakeword_model_paths: str = None,
                 openwakeword_inference_framework: str = "onnx",
                 wake_words: str = "",
                 wake_words_sensitivity: float = INIT_WAKE_WORDS_SENSITIVITY,
                 wake_word_activation_delay: float = (
                         INIT_WAKE_WORD_ACTIVATION_DELAY
                 ),
                 wake_word_timeout: float = INIT_WAKE_WORD_TIMEOUT,
                 wake_word_buffer_duration: float = INIT_WAKE_WORD_BUFFER_DURATION,
                 on_wakeword_detected=None,
                 on_wakeword_timeout=None,
                 on_wakeword_detection_start=None,
                 on_wakeword_detection_end=None,
                 on_recorded_chunk=None,
                 debug_mode=False,
                 handle_buffer_overflow: bool = INIT_HANDLE_BUFFER_OVERFLOW,
                 beam_size: int = 5,
                 beam_size_realtime: int = 3,
                 buffer_size: int = BUFFER_SIZE,
                 sample_rate: int = SAMPLE_RATE,
                 initial_prompt: Optional[Union[str, Iterable[int]]] = None,
                 suppress_tokens: Optional[List[int]] = [-1],
                 ):
        self.apikey = apikey
        self.language = language
        self.compute_type = compute_type
        self.input_device_index = input_device_index
        self.gpu_device_index = gpu_device_index
        self.device = device
        self.wake_words = wake_words
        self.wake_word_activation_delay = wake_word_activation_delay
        self.wake_word_timeout = wake_word_timeout
        self.wake_word_buffer_duration = wake_word_buffer_duration
        self.ensure_sentence_starting_uppercase = (
            ensure_sentence_starting_uppercase
        )
        self.ensure_sentence_ends_with_period = (
            ensure_sentence_ends_with_period
        )
        self.use_microphone = mp.Value(c_bool, use_microphone)
        self.min_gap_between_recordings = min_gap_between_recordings
        self.min_length_of_recording = min_length_of_recording
        self.pre_recording_buffer_duration = pre_recording_buffer_duration
        self.post_speech_silence_duration = post_speech_silence_duration
        self.on_recording_start = on_recording_start
        self.on_recording_stop = on_recording_stop
        self.on_wakeword_detected = on_wakeword_detected
        self.on_wakeword_timeout = on_wakeword_timeout
        self.on_vad_detect_start = on_vad_detect_start
        self.on_vad_detect_stop = on_vad_detect_stop
        self.on_wakeword_detection_start = on_wakeword_detection_start
        self.on_wakeword_detection_end = on_wakeword_detection_end
        self.on_recorded_chunk = on_recorded_chunk
        self.on_transcription_start = on_transcription_start
        self.enable_realtime_transcription = enable_realtime_transcription
        self.realtime_model_type = realtime_model_type
        self.realtime_processing_pause = realtime_processing_pause
        self.on_realtime_transcription_update = (
            on_realtime_transcription_update
        )
        self.on_realtime_transcription_stabilized = (
            on_realtime_transcription_stabilized
        )
        self.debug_mode = debug_mode
        self.handle_buffer_overflow = handle_buffer_overflow
        self.beam_size = beam_size
        self.beam_size_realtime = beam_size_realtime
        self.allowed_latency_limit = ALLOWED_LATENCY_LIMIT

        # --- NEW: Store new parameters ---
        self.voice_activity_threshold = voice_activity_threshold
        self.on_audio_level_update = on_audio_level_update

        self.level = level
        self.audio_queue = mp.Queue()
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.recording_start_time = 0
        self.recording_stop_time = 0
        self.wake_word_detect_time = 0
        self.silero_check_time = 0
        self.silero_working = False
        self.speech_end_silence_start = 0
        self.silero_sensitivity = silero_sensitivity
        self.listen_start = 0
        self.spinner = spinner
        self.halo = None
        self.state = "inactive"
        self.wakeword_detected = False
        self.text_storage = []
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.is_webrtc_speech_active = False
        self.is_silero_speech_active = False
        self.recording_thread = None
        self.realtime_thread = None
        self.audio_interface = None
        self.audio = None
        self.stream = None
        self.start_recording_event = threading.Event()
        self.stop_recording_event = threading.Event()
        self.last_transcription_bytes = None
        self.initial_prompt = initial_prompt
        self.suppress_tokens = suppress_tokens
        self.use_wake_words = wake_words or wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}

        # Initialize the logging configuration with the specified level
        log_format = 'YourOwnWaifu: %(name)s - %(levelname)s - %(message)s'

        # Create a logger
        logger = logging.getLogger()
        logger.setLevel(level)  # Set the root logger's level

        # Create a file handler and set its level
        file_handler = logging.FileHandler('files/app.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))

        # Create a console handler and set its level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(log_format))

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        self.is_shut_down = False
        self.shutdown_event = mp.Event()

        try:
            logging.debug("Explicitly setting the multiprocessing start method to 'spawn'")
            mp.set_start_method('spawn')
        except RuntimeError as e:
            logging.debug(f"Start method has already been set. Details: {e}")

        logging.info("Starting RealTimeSTT")

        self.interrupt_stop_event = mp.Event()
        self.was_interrupted = mp.Event()
        self.main_transcription_ready_event = mp.Event()
        self.parent_transcription_pipe, child_transcription_pipe = mp.Pipe()

        # Set device for model
        self.device =  "cpu"

        self.transcript_process = self._start_thread(
            target=AudioToTextRecorder._transcription_worker,
            args=(
                child_transcription_pipe,
                self.main_transcription_ready_event,
                self.shutdown_event,
                self.interrupt_stop_event,
                self.apikey
            )
        )

        # Start audio data reading process
        if self.use_microphone.value:
            logging.info("Initializing audio recording"
                         " (creating pyAudio input stream,"
                         f" sample rate: {self.sample_rate}"
                         f" buffer size: {self.buffer_size}"
                         )
            self.reader_process = self._start_thread(
                target=AudioToTextRecorder._audio_data_worker,
                args=(
                    self.audio_queue,
                    self.sample_rate,
                    self.buffer_size,
                    self.input_device_index,
                    self.shutdown_event,
                    self.interrupt_stop_event,
                    self.use_microphone
                )
            )

        try:
            logging.info("Initializing WebRTC voice with "
                         f"Sensitivity {webrtc_sensitivity}"
                         )
            self.webrtc_vad_model = webrtcvad.Vad()
            self.webrtc_vad_model.set_mode(webrtc_sensitivity)

        except Exception as e:
            logging.exception("Error initializing WebRTC voice "
                              f"activity detection engine: {e}"
                              )
            raise

        logging.debug("WebRTC VAD voice activity detection "
                      "engine initialized successfully"
                      )

        self.audio_buffer = collections.deque(
            maxlen=int((self.sample_rate // self.buffer_size) *
                       self.pre_recording_buffer_duration)
        )
        self.frames = []

        # Recording control flags
        self.is_recording = False
        self.is_running = True
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False

        # Start the recording worker thread
        self.recording_thread = threading.Thread(target=self._recording_worker)
        self.recording_thread.daemon = True
        self.recording_thread.start()

        # Start the realtime transcription worker thread
        self.realtime_thread = threading.Thread(target=self._realtime_worker)
        self.realtime_thread.daemon = True
        self.realtime_thread.start()

        # Wait for transcription models to start
        logging.debug('Waiting for main transcription model to start')
        self.main_transcription_ready_event.wait()
        logging.debug('Main transcription model ready')

        logging.debug('RealtimeSTT initialization completed successfully')

    def _start_thread(self, target=None, args=()):
        """
        Implement a consistent threading model across the library.

        This method is used to start any thread in this library. It uses the
        standard threading. Thread for Linux and for all others uses the pytorch
        MultiProcessing library 'Process'.
        Args:
            target (callable object): is the callable object to be invoked by
              the run() method. Defaults to None, meaning nothing is called.
            args (tuple): is a list or tuple of arguments for the target
              invocation. Defaults to ().
        """
        if (platform.system() == 'Linux'):
            thread = threading.Thread(target=target, args=args)
            thread.deamon = True
            thread.start()
            return thread
        else:
            thread = mp.Process(target=target, args=args)
            thread.start()
            return thread

    @staticmethod
    def _transcription_worker(conn,
                            ready_event,
                            shutdown_event,
                            interrupt_stop_event,
                            apikey
                            ):
        """
        Worker method that passes audio data to the message handler.
        """
        import logging
        import time
        import io

        logging.info("Initializing audio passthrough worker")
        ready_event.set()
        logging.debug("Audio passthrough worker initialized successfully")

        while not shutdown_event.is_set():
            try:
                if conn.poll(0.5):
                    audio, language = conn.recv()
                    try:
                        # Create a buffer and write the audio as MP3
                        buffer = io.BytesIO()
                        write(buffer, audio, 16000, format='MP3')
                        buffer.seek(0)

                        # Get the properly formatted MP3 bytes
                        audio_bytes = buffer.getvalue()

                        conn.send(('success', audio_bytes))
                        logging.debug("Successfully processed and sent audio data")
                    except Exception as e:
                        logging.error(f"Audio processing error: {e}")
                        conn.send(('error', str(e)))
                else:
                    time.sleep(0.02)

            except KeyboardInterrupt:
                interrupt_stop_event.set()
                logging.debug("Audio worker process finished due to KeyboardInterrupt")
                break
            except Exception as e:
                logging.error(f"Error during audio processing: {e}")
                continue

    @staticmethod
    def _audio_data_worker(audio_queue,
                           sample_rate,
                           buffer_size,
                           input_device_index,
                           shutdown_event,
                           interrupt_stop_event,
                           use_microphone):
        """
        Worker method that handles the audio recording process.

        This method runs in a separate process and is responsible for:
        - Setting up the audio input stream for recording.
        - Continuously reading audio data from the input stream
          and placing it in a queue.
        - Handling errors during the recording process, including
          input overflow.
        - Gracefully terminating the recording process when a shutdown
          event is set.

        Args:
            audio_queue (queue.Queue): A queue where recorded audio
              data is placed.
            sample_rate (int): The sample rate of the audio input stream.
            buffer_size (int): The size of the buffer used in the audio
              input stream.
            input_device_index (int): The index of the audio input device
            shutdown_event (threading.Event): An event that, when set, signals
              this worker method to terminate.

        Raises:
            Exception: If there is an error while initializing the audio
              recording.
        """
        try:
            audio_interface = pyaudio.PyAudio()
            if input_device_index is None:
                default_device = audio_interface.get_default_input_device_info()
                input_device_index = default_device['index']
            stream = audio_interface.open(
                rate=sample_rate,
                format=pyaudio.paInt16,
                channels=1,
                input=True,
                frames_per_buffer=buffer_size,
                input_device_index=input_device_index,
            )

        except Exception as e:
            logging.exception("Error initializing pyaudio "
                              f"audio recording: {e}"
                              )
            raise

        logging.debug("Audio recording (pyAudio input "
                      "stream) initialized successfully"
                      )

        try:
            while not shutdown_event.is_set():
                try:
                    data = stream.read(buffer_size)

                except OSError as e:
                    if e.errno == pyaudio.paInputOverflowed:
                        logging.warning("Input overflowed. Frame dropped.")
                    else:
                        logging.error(f"Error during recording: {e}")
                    tb_str = traceback.format_exc()
                    continue

                except Exception as e:
                    logging.error(f"Error during recording: {e}")
                    tb_str = traceback.format_exc()
                    continue

                if use_microphone.value:
                    audio_queue.put(data)

        except KeyboardInterrupt:
            interrupt_stop_event.set()
            logging.debug("Audio data worker process "
                          "finished due to KeyboardInterrupt"
                          )
        finally:
            stream.stop_stream()
            stream.close()
            audio_interface.terminate()

    def wakeup(self):
        """
        If in wake work modus, wake up as if a wake word was spoken.
        """
        self.listen_start = time.time()

    def abort(self):
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False
        self._set_state("inactive")
        self.interrupt_stop_event.set()
        self.was_interrupted.wait()
        self.was_interrupted.clear()

    def wait_audio(self):
        """
        Waits for the start and completion of the audio recording process.

        This method is responsible for:
        - Waiting for voice activity to begin recording if not yet started.
        - Waiting for voice inactivity to complete the recording.
        - Setting the audio buffer from the recorded frames.
        - Resetting recording-related attributes.

        Side effects:
        - Updates the state of the instance.
        - Modifies the audio attribute to contain the processed audio data.
        """

        self.listen_start = time.time()

        # If not yet started recording, wait for voice activity to initiate.
        if not self.is_recording and not self.frames:
            self._set_state("listening")
            self.start_recording_on_voice_activity = True

            # Wait until recording starts
            while not self.interrupt_stop_event.is_set():
                if self.start_recording_event.wait(timeout=0.02):
                    break

        # If recording is ongoing, wait for voice inactivity
        # to finish recording.
        if self.is_recording:
            self.stop_recording_on_voice_deactivity = True

            # Wait until recording stops
            while not self.interrupt_stop_event.is_set():
                if (self.stop_recording_event.wait(timeout=0.02)):
                    break

        # Convert recorded frames to the appropriate audio format.
        audio_array = np.frombuffer(b''.join(self.frames), dtype=np.int16)
        self.audio = audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE
        self.frames.clear()

        # Reset recording-related timestamps
        self.recording_stop_time = 0
        self.listen_start = 0

        self._set_state("inactive")

    def transcribe(self):
        """
        Transcribes audio captured by this class instance using the
        `faster_whisper` model.

        Automatically starts recording upon voice activity if not manually
          started using `recorder.start()`.
        Automatically stops recording upon voice deactivity if not manually
          stopped with `recorder.stop()`.
        Processes the recorded audio to generate transcription.

        Args:
            on_transcription_finished (callable, optional): Callback function
              to be executed when transcription is ready.
            If provided, transcription will be performed asynchronously,
              and the callback will receive the transcription as its argument.
              If omitted, the transcription will be performed synchronously,
              and the result will be returned.

        Returns (if no callback is set):
            str: The transcription of the recorded audio.

        Raises:
            Exception: If there is an error during the transcription process.
        """
        self._set_state("transcribing")
        audio_copy = copy.deepcopy(self.audio)
        self.parent_transcription_pipe.send((self.audio, self.language))
        status, result = self.parent_transcription_pipe.recv()

        self._set_state("inactive")
        if status == 'success':
            self.last_transcription_bytes = audio_copy
            return self._preprocess_output(result)
        else:
            logging.error(result)
            raise Exception(result)

    def _process_wakeword(self, data):
        """
        Processes audio data to detect wake words.
        """
        if self.wakeword_backend in {'pvp', 'pvporcupine'}:
            pcm = struct.unpack_from(
                "h" * self.buffer_size,
                data
            )
            porcupine_index = self.porcupine.process(pcm)
            if self.debug_mode:
                print(f"wake words porcupine_index: {porcupine_index}")
            return self.porcupine.process(pcm)

        elif self.wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}:
            pcm = np.frombuffer(data, dtype=np.int16)
            prediction = self.owwModel.predict(pcm)
            max_score = -1
            max_index = -1
            wake_words_in_prediction = len(self.owwModel.prediction_buffer.keys())
            self.wake_words_sensitivities
            if wake_words_in_prediction:
                for idx, mdl in enumerate(self.owwModel.prediction_buffer.keys()):
                    scores = list(self.owwModel.prediction_buffer[mdl])
                    if scores[-1] >= self.wake_words_sensitivity and scores[-1] > max_score:
                        max_score = scores[-1]
                        max_index = idx
                if self.debug_mode:
                    print(f"wake words oww max_index, max_score: {max_index} {max_score}")
                return max_index
            else:
                if self.debug_mode:
                    print(f"wake words oww_index: -1")
                return -1

        if self.debug_mode:
            print("wake words no match")
        return -1

    def text(self,
             on_transcription_finished=None,
             ):
        """
        Transcribes audio captured by this class instance
        using the `faster_whisper` model.

        - Automatically starts recording upon voice activity if not manually
          started using `recorder.start()`.
        - Automatically stops recording upon voice deactivity if not manually
          stopped with `recorder.stop()`.
        - Processes the recorded audio to generate transcription.

        Args:
            on_transcription_finished (callable, optional): Callback function
              to be executed when transcription is ready.
            If provided, transcription will be performed asynchronously, and
              the callback will receive the transcription as its argument.
              If omitted, the transcription will be performed synchronously,
              and the result will be returned.

        Returns (if not callback is set):
            str: The transcription of the recorded audio
        """

        self.interrupt_stop_event.clear()
        self.was_interrupted.clear()

        self.wait_audio()

        if self.is_shut_down or self.interrupt_stop_event.is_set():
            if self.interrupt_stop_event.is_set():
                self.was_interrupted.set()
            return ""

        if on_transcription_finished:
            threading.Thread(target=on_transcription_finished,
                             args=(self.transcribe(),)).start()
        else:
            return self.transcribe()

    def start(self):
        """
        Starts recording audio directly without waiting for voice activity.
        """

        # Ensure there's a minimum interval
        # between stopping and starting recording
        if (time.time() - self.recording_stop_time
                < self.min_gap_between_recordings):
            logging.info("Attempted to start recording "
                         "too soon after stopping."
                         )
            return self

        logging.info("recording started")
        self._set_state("recording")
        self.text_storage = []
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.wakeword_detected = False
        self.wake_word_detect_time = 0
        self.frames = []
        self.is_recording = True
        self.recording_start_time = time.time()
        self.is_silero_speech_active = False
        self.is_webrtc_speech_active = False
        self.stop_recording_event.clear()
        self.start_recording_event.set()

        if self.on_recording_start:
            self.on_recording_start()

        return self

    def stop(self):
        """
        Stops recording audio.
        """

        # Ensure there's a minimum interval
        # between starting and stopping recording
        if (time.time() - self.recording_start_time
                < self.min_length_of_recording):
            logging.info("Attempted to stop recording "
                         "too soon after starting."
                         )
            return self

        logging.info("recording stopped")
        self.is_recording = False
        self.recording_stop_time = time.time()
        self.is_silero_speech_active = False
        self.is_webrtc_speech_active = False
        self.silero_check_time = 0
        self.start_recording_event.clear()
        self.stop_recording_event.set()

        if self.on_recording_stop:
            self.on_recording_stop()

        return self

    def feed_audio(self, chunk, original_sample_rate=16000):
        """
        Feed an audio chunk into the processing pipeline. Chunks are
        accumulated until the buffer size is reached, and then the accumulated
        data is fed into the audio_queue.
        """
        # Check if the buffer attribute exists, if not, initialize it
        if not hasattr(self, 'buffer'):
            self.buffer = bytearray()

        # Check if input is a NumPy array
        if isinstance(chunk, np.ndarray):
            # Handle stereo to mono conversion if necessary
            if chunk.ndim == 2:
                chunk = np.mean(chunk, axis=1)

            # Resample to 16000 Hz if necessary
            if original_sample_rate != 16000:
                num_samples = int(len(chunk) * 16000 / original_sample_rate)
                chunk = resample(chunk, num_samples)

            # Ensure data type is int16
            chunk = chunk.astype(np.int16)

            # Convert the NumPy array to bytes
            chunk = chunk.tobytes()

        # Append the chunk to the buffer
        self.buffer += chunk
        buf_size = 2 * self.buffer_size  # silero complains if too short

        # Check if the buffer has reached or exceeded the buffer_size
        while len(self.buffer) >= buf_size:
            # Extract self.buffer_size amount of data from the buffer
            to_process = self.buffer[:buf_size]
            self.buffer = self.buffer[buf_size:]

            # Feed the extracted data to the audio_queue
            self.audio_queue.put(to_process)

    def set_microphone(self, microphone_on=True):
        """
        Set the microphone on or off.
        """
        logging.info("Setting microphone to: " + str(microphone_on))
        self.use_microphone.value = microphone_on

    def shutdown(self):
        """
        Safely shuts down the audio recording by stopping the
        recording worker and closing the audio stream.
        """

        # Force wait_audio() and text() to exit
        self.is_shut_down = True
        self.start_recording_event.set()
        self.stop_recording_event.set()

        self.shutdown_event.set()
        self.is_recording = False
        self.is_running = False

        logging.debug('Finishing recording thread')
        if self.recording_thread:
            self.recording_thread.join()

        logging.debug('Terminating reader process')

        # Give it some time to finish the loop and cleanup.
        if self.use_microphone:
            self.reader_process.join(timeout=10)

        if self.reader_process.is_alive():
            logging.warning("Reader process did not terminate "
                            "in time. Terminating forcefully."
                            )
            self.reader_process.terminate()

        logging.debug('Terminating transcription process')
        self.transcript_process.join(timeout=10)

        if self.transcript_process.is_alive():
            logging.warning("Transcript process did not terminate "
                            "in time. Terminating forcefully."
                            )
            self.transcript_process.terminate()

        self.parent_transcription_pipe.close()

        logging.debug('Finishing realtime thread')
        if self.realtime_thread:
            self.realtime_thread.join()

        if self.enable_realtime_transcription:
            if self.realtime_model_type:
                del self.realtime_model_type
                self.realtime_model_type = None
        gc.collect()

    def _recording_worker(self):
        """
        The main worker method which constantly monitors the audio
        input for voice activity and accordingly starts/stops the recording.
        """
        logging.debug('Starting recording worker')
        try:
            was_recording = False
            delay_was_passed = False
            while self.is_running:
                try:
                    data = self.audio_queue.get()
                    if self.on_recorded_chunk:
                        self.on_recorded_chunk(data)
                    if self.handle_buffer_overflow and self.audio_queue.qsize() > self.allowed_latency_limit:
                        logging.warning(f"Audio queue size ({self.audio_queue.qsize()}) exceeds limit. Discarding chunks.")
                        while self.audio_queue.qsize() > self.allowed_latency_limit:
                            data = self.audio_queue.get()
                except BrokenPipeError:
                    self.is_running = False
                    break

                # --- NEW: Centralized audio level calculation and callback ---
                # This is the most important change. We do this for EVERY chunk.
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                level = np.max(np.abs(audio_chunk)) if len(audio_chunk) > 0 else 0
                if self.on_audio_level_update:
                    self.on_audio_level_update(int(level))
                # --- End of new section ---

                if not self.is_recording:
                    time_since_listen_start = (time.time() - self.listen_start if self.listen_start else 0)
                    wake_word_activation_delay_passed = (time_since_listen_start > self.wake_word_activation_delay)
                    if wake_word_activation_delay_passed and not delay_was_passed:
                        if self.use_wake_words and self.wake_word_activation_delay and self.on_wakeword_timeout:
                            self.on_wakeword_timeout()
                    delay_was_passed = wake_word_activation_delay_passed

                    if not self.recording_stop_time:
                        if self.use_wake_words and wake_word_activation_delay_passed and not self.wakeword_detected:
                            self._set_state("wakeword")
                        else:
                            self._set_state("listening" if self.listen_start else "inactive")

                    if self.use_wake_words and wake_word_activation_delay_passed:
                        try:
                            wakeword_index = self._process_wakeword(data)
                            if wakeword_index >= 0:
                                samples_time = int(self.sample_rate * self.wake_word_buffer_duration)
                                start_index = max(0, len(self.audio_buffer) - samples_time)
                                temp_samples = collections.deque(itertools.islice(self.audio_buffer, start_index, None))
                                self.audio_buffer.clear()
                                self.audio_buffer.extend(temp_samples)
                                self.wake_word_detect_time = time.time()
                                self.wakeword_detected = True
                                if self.on_wakeword_detected:
                                    self.on_wakeword_detected()
                        except Exception as e:
                            logging.error(f"Wake word processing error: {e}")
                            continue

                    if ((not self.use_wake_words or not wake_word_activation_delay_passed) and self.start_recording_on_voice_activity) or self.wakeword_detected:
                        # --- MODIFIED: Pass the calculated level to the check ---
                        if self._is_voice_active(data, level):
                            logging.info("voice activity detected")
                            self.start()
                            if self.is_recording:
                                self.start_recording_on_voice_activity = False
                                self.frames.extend(list(self.audio_buffer))
                                self.audio_buffer.clear()
                        else:
                            # The check is already done inside _is_voice_active
                            pass
                    self.speech_end_silence_start = 0
                else: # If we are currently recording
                    if self.stop_recording_on_voice_deactivity:
                        # --- MODIFIED: Use a simpler check now ---
                        is_speech = self._is_webrtc_speech(data, True)
                        if not is_speech:
                            if self.speech_end_silence_start == 0:
                                self.speech_end_silence_start = time.time()
                        else:
                            self.speech_end_silence_start = 0

                        if self.speech_end_silence_start and (time.time() - self.speech_end_silence_start > self.post_speech_silence_duration):
                            logging.info("voice deactivity detected")
                            self.stop()

                if not self.is_recording and was_recording:
                    self.stop_recording_on_voice_deactivity = False

                if self.wake_word_detect_time and (time.time() - self.wake_word_detect_time > self.wake_word_timeout):
                    self.wake_word_detect_time = 0
                    if self.wakeword_detected and self.on_wakeword_timeout:
                        self.on_wakeword_timeout()
                    self.wakeword_detected = False

                was_recording = self.is_recording
                if self.is_recording:
                    self.frames.append(data)
                if not self.is_recording or self.speech_end_silence_start:
                    self.audio_buffer.append(data)
        except Exception as e:
            if not self.interrupt_stop_event.is_set():
                logging.error(f"Unhandled exception in _recording_worker: {e}")
                raise

    def _realtime_worker(self):
        """
        Performs real-time transcription if the feature is enabled.

        The method is responsible transcribing recorded audio frames
          in real-time based on the specified resolution interval.
        The transcribed text is stored in `self.realtime_transcription_text`
          and a callback
        function is invoked with this text if specified.
        """

        try:

            logging.debug('Starting realtime worker')

            # Return immediately if real-time transcription is not enabled
            if not self.enable_realtime_transcription:
                return

            # Continue running as long as the main process is active
            while self.is_running:

                # Check if the recording is active
                if self.is_recording:

                    # Sleep for the duration of the transcription resolution
                    time.sleep(self.realtime_processing_pause)

                    # Convert the buffer frames to a NumPy array
                    audio_array = np.frombuffer(
                        b''.join(self.frames),
                        dtype=np.int16
                    )

                    # Normalize the array to a [-1, 1] range

                    # double check recording state
                    # because it could have changed mid-transcription
                    if self.is_recording and time.time() - \
                            self.recording_start_time > 0.5:

                        logging.debug('Starting realtime transcription')

                        self.realtime_transcription_text = ""

                        self.text_storage.append(
                            self.realtime_transcription_text
                        )

                        # Take the last two texts in storage, if they exist
                        if len(self.text_storage) >= 2:
                            last_two_texts = self.text_storage[-2:]

                            # Find the longest common prefix
                            # between the two texts
                            prefix = os.path.commonprefix(
                                [last_two_texts[0], last_two_texts[1]]
                            )

                            # This prefix is the text that was transcripted
                            # two times in the same way
                            # Store as "safely detected text"
                            if len(prefix) >= \
                                    len(self.realtime_stabilized_safetext):
                                # Only store when longer than the previous
                                # as additional security
                                self.realtime_stabilized_safetext = prefix

                        # Find parts of the stabilized text
                        # in the freshly transcripted text
                        matching_pos = self._find_tail_match_in_text(
                            self.realtime_stabilized_safetext,
                            self.realtime_transcription_text
                        )

                        if matching_pos < 0:
                            if self.realtime_stabilized_safetext:
                                self._on_realtime_transcription_stabilized(
                                    self._preprocess_output(
                                        self.realtime_stabilized_safetext,
                                        True
                                    )
                                )
                            else:
                                self._on_realtime_transcription_stabilized(
                                    self._preprocess_output(
                                        self.realtime_transcription_text,
                                        True
                                    )
                                )
                        else:
                            # We found parts of the stabilized text
                            # in the transcripted text
                            # We now take the stabilized text
                            # and add only the freshly transcripted part to it
                            output_text = self.realtime_stabilized_safetext + \
                                          self.realtime_transcription_text[matching_pos:]

                            # This yields us the "left" text part as stabilized
                            # AND at the same time delivers fresh detected
                            # parts on the first run without the need for
                            # two transcriptions
                            self._on_realtime_transcription_stabilized(
                                self._preprocess_output(output_text, True)
                            )

                        # Invoke the callback with the transcribed text
                        self._on_realtime_transcription_update(
                            self._preprocess_output(
                                self.realtime_transcription_text,
                                True
                            )
                        )

                # If not recording, sleep briefly before checking again
                else:
                    time.sleep(TIME_SLEEP)

        except Exception as e:
            logging.error(f"Unhandled exeption in _realtime_worker: {e}")
            raise

    def _is_silero_speech(self, level):
        """
        MODIFIED: This function now only checks the pre-calculated level against the threshold.
        The UI callback has been moved out.
        """
        is_speech_active = level > self.voice_activity_threshold
        if self.debug_mode:
            logging.debug(f"Level: {level}, Threshold: {self.voice_activity_threshold}, Detected: {is_speech_active}")
        if is_speech_active:
            self.is_silero_speech_active = True
        return is_speech_active

    def _is_webrtc_speech(self, chunk, all_frames_must_be_true=False):
        """
        Returns true if speech is detected in the provided audio data

        Args:
            data (bytes): raw bytes of audio data (1024 raw bytes with
            16000 sample rate and 16 bits per sample)
        """
        if self.sample_rate != 16000:
            pcm_data = np.frombuffer(chunk, dtype=np.int16)
            data_16000 = signal.resample_poly(
                pcm_data, 16000, self.sample_rate)
            chunk = data_16000.astype(np.int16).tobytes()

        # Number of audio frames per millisecond
        frame_length = int(16000 * 0.01)  # for 10ms frame
        num_frames = int(len(chunk) / (2 * frame_length))
        speech_frames = 0

        for i in range(num_frames):
            start_byte = i * frame_length * 2
            end_byte = start_byte + frame_length * 2
            frame = chunk[start_byte:end_byte]
            if self.webrtc_vad_model.is_speech(frame, 16000):
                speech_frames += 1
                if not all_frames_must_be_true:
                    if self.debug_mode:
                        print(f"Speech detected in frame {i + 1}"
                              f" of {num_frames}")
                    return True
        if all_frames_must_be_true:
            if self.debug_mode and speech_frames == num_frames:
                print(f"Speech detected in {speech_frames} of "
                      f"{num_frames} frames")
            elif self.debug_mode:
                print(f"Speech not detected in all {num_frames} frames")
            return speech_frames == num_frames
        else:
            if self.debug_mode:
                print(f"Speech not detected in any of {num_frames} frames")
            return False

    def _check_voice_activity(self, data, level):
        """
        MODIFIED: Accepts the pre-calculated level.
        """
        self.is_webrtc_speech_active = self._is_webrtc_speech(data)
        if self.is_webrtc_speech_active:
            # Now we check the amplitude level
            self._is_silero_speech(level)

    def _is_voice_active(self, data, level):
        """
        MODIFIED: This is now the main VAD trigger function.
        It performs both checks.
        """
        self._check_voice_activity(data, level)
        return self.is_webrtc_speech_active and self.is_silero_speech_active

    def _set_state(self, new_state):
        """
        Update the current state of the recorder and execute
        corresponding state-change callbacks.

        Args:
            new_state (str): The new state to set.

        """
        # Check if the state has actually changed
        if new_state == self.state:
            return

        # Store the current state for later comparison
        old_state = self.state

        # Update to the new state
        self.state = new_state

        # Execute callbacks based on transitioning FROM a particular state
        if old_state == "listening":
            if self.on_vad_detect_stop:
                self.on_vad_detect_stop()
        elif old_state == "wakeword":
            if self.on_wakeword_detection_end:
                self.on_wakeword_detection_end()

        # Execute callbacks based on transitioning TO a particular state
        if new_state == "listening":
            if self.on_vad_detect_start:
                self.on_vad_detect_start()
            self._set_spinner("speak now")
            if self.spinner and self.halo:
                self.halo._interval = 250
        elif new_state == "wakeword":
            if self.on_wakeword_detection_start:
                self.on_wakeword_detection_start()
            self._set_spinner(f"say {self.wake_words}")
            if self.spinner and self.halo:
                self.halo._interval = 500
        elif new_state == "transcribing":
            if self.on_transcription_start:
                self.on_transcription_start()
            self._set_spinner("transcribing")
            if self.spinner and self.halo:
                self.halo._interval = 50
        elif new_state == "recording":
            self._set_spinner("recording")
            if self.spinner and self.halo:
                self.halo._interval = 100
        elif new_state == "inactive":
            if self.spinner and self.halo:
                self.halo.stop()
                self.halo = None

    def _set_spinner(self, text):
        """
        Update the spinner's text or create a new
        spinner with the provided text.

        Args:
            text (str): The text to be displayed alongside the spinner.
        """
        if self.spinner:
            # If the Halo spinner doesn't exist, create and start it
            if self.halo is None:
                self.halo = halo.Halo(text=text)
                self.halo.start()
            # If the Halo spinner already exists, just update the text
            else:
                self.halo.text = text

    def _preprocess_output(self, text, preview=False):
        """
        Preprocesses the output text by removing any leading or trailing
        whitespace, converting all whitespace sequences to a single space
        character, and capitalizing the first character of the text.

        Args:
            text (str): The text to be preprocessed.

        Returns:
            str: The preprocessed text.
        """
        return text

    def _find_tail_match_in_text(self, text1, text2, length_of_match=10):
        """
        Find the position where the last 'n' characters of text1
        match with a substring in text2.

        This method takes two texts, extracts the last 'n' characters from
        text1 (where 'n' is determined by the variable 'length_of_match'), and
        searches for an occurrence of this substring in text2, starting from
        the end of text2 and moving towards the beginning.

        Parameters:
        - text1 (str): The text containing the substring that we want to find
          in text2.
        - text2 (str): The text in which we want to find the matching
          substring.
        - length_of_match(int): The length of the matching string that we are
          looking for

        Returns:
        int: The position (0-based index) in text2 where the matching
          substring starts. If no match is found or either of the texts is
          too short, returns -1.
        """

        # Check if either of the texts is too short
        if len(text1) < length_of_match or len(text2) < length_of_match:
            return -1

        # The end portion of the first text that we want to compare
        target_substring = text1[-length_of_match:]

        # Loop through text2 from right to left
        for i in range(len(text2) - length_of_match + 1):
            # Extract the substring from text2
            # to compare with the target_substring
            current_substring = text2[len(text2) - i - length_of_match:
                                      len(text2) - i]

            # Compare the current_substring with the target_substring
            if current_substring == target_substring:
                # Position in text2 where the match starts
                return len(text2) - i

        return -1

    def _on_realtime_transcription_stabilized(self, text):
        """
        Callback method invoked when the real-time transcription stabilizes.

        This method is called internally when the transcription text is
        considered "stable" meaning it's less likely to change significantly
        with additional audio input. It notifies any registered external
        listener about the stabilized text if recording is still ongoing.
        This is particularly useful for applications that need to display
        live transcription results to users and want to highlight parts of the
        transcription that are less likely to change.

        Args:
            text (str): The stabilized transcription text.
        """
        if self.on_realtime_transcription_stabilized:
            if self.is_recording:
                self.on_realtime_transcription_stabilized(text)

    def _on_realtime_transcription_update(self, text):
        """
        Callback method invoked when there's an update in the real-time
        transcription.

        This method is called internally whenever there's a change in the
        transcription text, notifying any registered external listener about
        the update if recording is still ongoing. This provides a mechanism
        for applications to receive and possibly display live transcription
        updates, which could be partial and still subject to change.

        Args:
            text (str): The updated transcription text.
        """
        if self.on_realtime_transcription_update:
            if self.is_recording:
                self.on_realtime_transcription_update(text)

    def __enter__(self):
        """
        Method to setup the context manager protocol.

        This enables the instance to be used in a `with` statement, ensuring
        proper resource management. When the `with` block is entered, this
        method is automatically called.

        Returns:
            self: The current instance of the class.
        """
        return self

    def __exit__(self):
        self.shutdown()