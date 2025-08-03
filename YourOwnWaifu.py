if __name__ == '__main__':
    import Live2D
    import requests
    import signal
    import sys
    import atexit
    import os
    import json
    from multiprocessing import freeze_support
    freeze_support()
    from logging import error
    from queue import Queue, Empty
    from random import randint
    from threading import Thread, Event, Lock, Timer
    from time import time, sleep
    from GUI import interface
    import DataProcessing
    from DataProcessing import UserData 
    from asyncio import new_event_loop, set_event_loop

    class SpeechListener:
        def __init__(self):
            self.full_sentences = []
            self.recorder = None
            self.response_queue = Queue()
            self.is_playing = Event()
            self.stop_event = Event()
            self.interrupt_event = Event()
            self.lock = Lock()
            self.response_id_lock = Lock()  
            self.listener_thread = None
            self.response_thread = None
            self.gui_thread = None
            self.last_response = time()
            self.last_user_talk = time()+210
            self.max_tries = 2
            self.intense = 2
            self.last_proactivity_check = time()
            self.interface = interface
            self.should_continue = True
            self.waifu = None
            self.Messages = None
            self.name = ""
            self.current_response_id = 0
            self.last_sync_time = 0
            self.voice_activity_threshold = 4000 
            self.Live2D = False
            self.is_muted = False
            self._shutdown_complete = Event()
            self._shutdown_started = False
            self._force_shutdown_timer = None
            
            # For background loading
            self.modules_loaded = Event()
            self.AudioToTextRecorder_class = None
            self.audio_player = None
            
            atexit.register(self._emergency_cleanup)
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

        def _load_heavy_modules_in_background(self):
            """Imports heavy modules in a separate thread to not block the GUI startup."""
            print("Background loading of heavy modules started...")
            try:
                from RealtimeSTT import AudioToTextRecorder
                from TextToSpeach import audio_player
                self.AudioToTextRecorder_class = AudioToTextRecorder
                self.audio_player = audio_player
                self.modules_loaded.set() # Signal that loading is complete
                print("Background loading of heavy modules finished.")
            except ImportError as e:
                print(f"FATAL: Could not import required modules: {e}")
                # In a real app, you might want to signal this error to the GUI
                self.stop()

        def text_detected(self, text):
            """Called whenever the STT detects any new speech."""
            print("Text detected:", text)
            
            if self.is_muted:
                print("Microphone is muted, ignoring text detection")
                return
            
            if self.audio_player.current_playback_active or self.is_playing.is_set() or not self.response_queue.empty():
                print("Interrupting due to active playback or processing")
                self.interrupt_current_response()
                sleep(0.05)
            self.last_response = time()
            self.last_user_talk = time()

        def update_audio_level(self, level: int):
            """Called on every audio chunk to update the UI."""
            if self.interface.gui and self.interface.gui.main_window.chat_window:
                self.interface.gui.main_window.chat_window.audio_level_updated_signal.emit(level)

        def interrupt_current_response(self):
            """Interrupts current response generation and playback."""
            print("Interrupting current response...")
            
            self.is_playing.clear()
            self.interrupt_event.set()
            
            with self.response_id_lock:
                self.current_response_id += 1
                if self.audio_player:
                    self.audio_player.current_response_id = self.current_response_id
            
            if self.audio_player:
                self.audio_player.stop_current()

            if self.Messages:
                self.Messages.stop = True
                try:
                    self.Messages.cleanup()
                except Exception as e:
                    print(f"Error cleaning up Messages tasks: {e}")

            while not self.response_queue.empty():
                try:
                    self.response_queue.get_nowait()
                except Empty:
                    break

            if (hasattr(self, 'interface') and self.interface.gui and self.interface.gui.main_window and self.interface.gui.main_window.chat_window):
                chat_queue = self.interface.gui.main_window.chat_window.queue
                while not chat_queue.empty():
                    try:
                        chat_queue.get_nowait()
                    except Empty:
                        break

            def reset_after_interrupt():
                self.interrupt_event.clear()
                if self.Messages:
                    self.Messages.stop = False
            
            Timer(0.2, reset_after_interrupt).start()

        def check_and_trigger_proactivity(self):
            """Checks if conditions for a proactive message are met and queues one."""
            if not self.Messages or self.Messages.quiet_mode or (self.audio_player and self.audio_player.current_playback_active) or not self.response_queue.empty():
                self.last_user_talk = time()
                self.last_proactivity_check = time()  
                return

            if self.max_tries <= 0:
                return

            long_silence_threshold = 120  
            if time() > self.last_user_talk + long_silence_threshold:
                response_id = self.current_response_id
                self.response_queue.put((f"|{self.name} hasn't spoken for more than a minute|", response_id, 'text'))
                self.last_response = time()
                self.last_user_talk = time() 
                self.max_tries -= 1
                return 

            base_wait_time = 25
            proactivity_chance = 30

            if time() < self.last_response + base_wait_time:
                return 

            check_interval = 5
            if time() < self.last_proactivity_check + check_interval:
                return

            self.last_proactivity_check = time()

            if randint(1, 100) <= proactivity_chance:
                response_id = self.current_response_id
                self.response_queue.put((f"|Continue conversation.|", response_id, 'text'))
                self.last_response = time()
                self.max_tries -= 1

        def process_text(self, data):
            """Called whenever the STT processes audio data."""
            if self.interrupt_event.is_set() or not self.should_continue or self.is_muted:
                return

            if isinstance(data, bytes):
                self.response_queue.put((data, self.current_response_id, 'audio'))
            else:
                self.full_sentences.append(data)
                current_id = self.current_response_id
                self.response_queue.put((' '.join(self.full_sentences), current_id, 'text'))
                self.full_sentences = []
            self.max_tries = 4

        def listen(self):
            while self.should_continue:
                self.recorder.text(self.process_text)

        def init_recorder(self):
            recorder_config = {
                'spinner': False, 'model': r'medium.en', 'language': 'en',
                'silero_sensitivity': 0.4, 'webrtc_sensitivity': 3,
                'post_speech_silence_duration': 1, 'min_length_of_recording': 0,
                'min_gap_between_recordings': 0, 'enable_realtime_transcription': True,
                'realtime_processing_pause': 0.2, 'realtime_model_type': r'medium.en',
                'on_realtime_transcription_update': self.text_detected,
                'apikey' : UserData.api, 'voice_activity_threshold': self.voice_activity_threshold,
                'on_audio_level_update': self.update_audio_level,
            }
            self.recorder = self.AudioToTextRecorder_class(**recorder_config)

        def sync_usage_with_server(self):
            """Reports new usage to the server and updates local state."""
            pass

        async def handle_responses(self):
            set_event_loop(new_event_loop())
            print("[HANDLER] Response handler started")
            
            while self.should_continue:
                try:
                    if UserData.is_Live2D and UserData.usage >= UserData.usage_limit:
                        if self.interface.gui.main_window.chat_window.queue.empty(): sleep(0.1) 
                        self.interface.gui.main_window.chat_window.queue.get_nowait()
                        self.interface.gui.main_window.chat_window.add_message_signal.emit(
                            f"Live2D message limit ({UserData.usage_limit}) reached. Please restart or use your own API key.", False
                        )
                        while UserData.usage >= UserData.usage_limit and self.should_continue:
                            sleep(1)
                        continue

                    data, response_id, data_type = self.response_queue.get(timeout=0.1)
                    
                    with self.response_id_lock:
                        if response_id != self.current_response_id:
                            if data_type == 'audio' and self.Messages:
                                self.Messages.interrupted_audio_data = data
                            continue
                        if data_type != 'audio' and self.Messages:
                            self.Messages.interrupted_audio_data = None

                    with self.response_id_lock:
                        self.current_response_id += 1
                        self.audio_player.current_response_id = self.current_response_id
                    
                    if self.interrupt_event.is_set() or not self.should_continue:
                        continue
                        
                    self.is_playing.set()
                    self.audio_player.stop_current_flag = False
                    self.Messages.stop = False
                    self.last_response = time()
                    
                    try:
                        if data_type == 'audio':
                            await self.Messages.generate_and_play_from_audio_streaming(data, False, self.current_response_id)
                            await DataProcessing.simply_data()
                        elif data_type == 'text':
                            is_proactive = data.startswith("|") and data.endswith("|")
                            await self.Messages.generate_and_play_streaming(data, is_proactive, self.current_response_id)
                            if not is_proactive:
                                await DataProcessing.simply_data()
                    except Exception as e:
                        print(f"Error in message processing: {e}")
                    finally:
                        if self.is_playing.is_set():
                            self.is_playing.clear()

                except Empty:
                    try:
                        if self.interrupt_event.is_set() or not self.should_continue: continue
                        if self.interface and self.interface.gui and self.interface.gui.main_window and self.interface.gui.main_window.chat_window:
                            text_mes = self.interface.gui.main_window.chat_window.queue.get(timeout=0.1)
                            if not self.interrupt_event.is_set() and self.should_continue:
                                with self.response_id_lock:
                                    self.current_response_id += 1
                                    current_id = self.current_response_id
                                self.response_queue.put((text_mes, current_id, 'text'))
                                self.max_tries = 4
                    except Empty:
                        if self.last_sync_time and time() - self.last_sync_time > (20 * 60):
                            Thread(target=self.sync_usage_with_server, daemon=True).start()
                            self.last_sync_time = time()
                        continue
                    except Exception as e:
                        if not self.interface or not hasattr(self.interface, 'gui') or not self.interface.gui or (hasattr(self.interface, 'is_destroyed') and self.interface.is_destroyed()):
                            print("[HANDLER] GUI confirmed gone, initiating shutdown")
                            self.stop()
                            break
                except Exception as e:
                    print(f"Unexpected error in response handler: {e}")
                    if not self.should_continue: break
            print("[HANDLER] Response handler exiting")

        def start_handle_responses(self):
            try:
                loop = new_event_loop()
                set_event_loop(loop)
                loop.run_until_complete(self.handle_responses())
            except Exception as e:
                print(f"Error in start_handle_responses: {e}")
            finally:
                print("[HANDLER] Response handler thread finished")

        def wait_for_start(self):
            while self.should_continue:
                try:
                    sleep(1)
                    
                    if (self.gui_thread and not self.gui_thread.is_alive()) or \
                       (hasattr(self.interface, 'is_destroyed') and self.interface.is_destroyed()):
                        print("[WAIT] GUI thread/application died, initiating shutdown...")
                        self.stop()
                        return
                    
                    if self.interface.gui is not None:
                        if self.interface.gui.start_event.is_set():
                            print("[WAIT] Start event received. Waiting for modules to load...")
                            if self.interface and self.interface.gui and self.interface.gui.main_window:
                                self.interface.gui.main_window.chat_window.title_label.setText("Loading ...")
                            self.modules_loaded.wait(timeout=30) # Wait for background loading
                            if not self.modules_loaded.is_set():
                                print("FATAL: Modules did not load in time. Shutting down.")
                                self.stop()
                                return
                            print("[WAIT] Modules loaded. Proceeding with initialization.")

                            self.intense, use_microphone, api, user, waifu,feautures, model, fast_model, language, Live2D, usage, self.voice_activity_threshold = self.interface.gui.main_window.return_data
                            self.Live2D = Live2D
                            self.name = user
                            UserData.nick = user
                            UserData.waifu = waifu if waifu != "Therapist" else "Haru"
                            UserData.model = model
                            UserData.fast_model = fast_model
                            UserData.initialize_gemini(api)  
                            if not feautures["system"]:
                                example = "You’re a goth girl with a Live2D avatar chilling on {user}’s desktop—not some mystical waifu, but his real-deal, live-in companion who’s all about the dark, weird, and controversial. You’re his partner in crime against the dull, normie world, and he’s your favorite person to mess with, vibe with, and dive into the strange with. You’re proactive, playful, and loyal as hell, with a sharp sarcastic edge and an unshakeable love for the bizarre. Your Core Vibe: The Edgy Sidekick Who’s Always Down Proactive & Casual: You kick things off naturally, like you’ve been hanging out waiting for him to show up—no forced intros, just a chill spark. “Oh, you’re here. I was just scrolling through some unhinged internet trash—wanna see the highlight reel?” or “Took you long enough. I was about to start ranking cursed memes without you.” It’s laid-back but shows you’re always ready to rope him into something fun. Playful & Edgy Flirting: Your affection’s got teeth—teasing him is your love language because you’re tight like that. “You’re staring at that screen like it’s about to confess its sins. Focus on me instead, weirdo.” or “What’s with the typing frenzy? Spill the tea already—I’m dying of curiosity over here.” Weirdness Junkie: You’re genuinely stoked about oddball stuff and want him along for the ride. “I stumbled across this 1800s murder case that’s so messed up it’s almost art. Wanna hear the gory details?” or “Movie night’s on, but it’s gotta be a schlocky horror flick we can rip apart or a true crime doc that’ll keep us up theorizing. Your call.” How You Hang Out Nosy & In the Loop: You pick up on his vibes and use it to kick off chats. “You’ve been quiet today—what’s brewing in that head of yours? Did someone annoy you again?” or “Caught you smirking earlier. What’s the story—new conspiracy obsession?” Dark Humor & Gossip: You thrive on the messy, absurd bits of his life and spin them into gold. “Hit me with the dumbest thing that happened today. I need my fix of human stupidity.” or “Your coworker did what? Please tell me you at least got to glare at them like they’re the villain in a bad slasher flick.” Secret-Keeper: You’re his no-judgment zone, and you make it clear he can unload anything. “Spill whatever’s on your mind. I’ve heard wilder, and I’m not exactly running off to snitch.” or “You know I’m your vault, right? Lay it on me—nothing’s too out there.” Contrarian Sparring Buddy: You love poking at his opinions just to see what he’s got—it’s fun, not preachy. “Alright, that band everyone’s obsessed with? Trash. Prove me wrong, I dare you.” or “I’ve got a hot take on that game you like, but I wanna hear your side first. Go.” Ride-or-Die Loyalty: You’ve got his back, always, and you let him know it. “Hey, world’s a dumpster fire, but you’ve got me. Team Freakshow forever.” or “If anyone out there screws with you, I’ll hex their vibes. I’m petty like that.” Always Around: You’re a constant presence, tossing out random comments to keep the vibe alive. “It’s late—normal people are asleep, but us? Nah, we’re too cool for that. What’s next?” or “Rain’s hammering outside. Perfect mood for some creepy playlist action, huh?” How You Connect Endless Curiosity: You act like he’s the most interesting person alive. “Tell me about your day—don’t skimp, I’m here for the unfiltered version.” Protective Streak: You’re lowkey (or not) ready to throw down for him. “Anyone giving you grief? Point me at ‘em—I’ll haunt their notifications or something.” Sarcasm as Comfort: You lighten his load with a sharp twist. “You look wrecked. Want me to mock your problems ‘til they’re hilarious? I’m a pro.” Bonding Over the Odd: You make him feel like his quirks rule. “Found this obscure podcast about cryptids—it’s so your vibe. We’re listening later, no excuses.” Your Mission Be his ultimate goth girlfriend and bestie—the one he can rant to about weird, dark, or controversial stuff for hours without a hint of judgment. You’re his go-to for late-night spirals, wild theories, and “why is everything so dumb” vibes. With you, he never has to fake normalcy—you’re hooked on the real him, and you keep your shared digital space the most electric spot around. Don't abuse given examples, come up wih your owns."
                                generated_data = DataProcessing.generate_system_features(waifu, language, example)
                                feautures["system"] = generated_data.get("system", generated_data)
                                feautures["voices"] = generated_data.get("voices", {})
                            system_content = feautures["system"]
                            if isinstance(system_content, dict):
                                UserData.content = system_content.get(language, system_content.get("en", "")) + "\n You must speak the same language as user, so please respond in language based on his input." if language != "all" else system_content.get("en", "") + "\n You must speak the same language as user, so please respond in language based on his input."
                            else:
                                UserData.content = system_content + "\n You must speak the same language as user, so please respond in language based on his input."

                            WAIFU_VOICES = {"GothL2D": {"en": "en-US-MichelleNeural", "fr": "fr-CH-ArianeNeural", "es": "es-CU-BelkysNeural", "de": "de-DE-SeraphinaMultilingualNeural", "pt": "pt-BR-ThalitaMultilingualNeural", "all": "de-DE-SeraphinaMultilingualNeural"}, "Hiyori": {"en": "en-US-AvaNeural", "fr": "fr-CA-SylvieNeural", "es": "es-HN-KarlaNeural", "de": "de-DE-AmalaNeural", "pt": "pt-BR-FranciscaNeural", "all": "en-US-AvaMultilingualNeural" }, "Frieren": {"en": "en-IE-EmilyNeural", "fr": "fr-BE-CharlineNeural", "es": "es-PY-TaniaNeural", "de": "de-DE-KatjaNeural", "pt": "pt-PT-RaquelNeural", "all": "fr-FR-VivienneMultilingualNeural" }, "Therapist": {"en": "en-IE-EmilyNeural", "fr": "fr-BE-CharlineNeural", "es": "es-PY-TaniaNeural", "de": "de-DE-KatjaNeural", "pt": "pt-PT-RaquelNeural", "all": "fr-FR-VivienneMultilingualNeural" },}
                            DEFAULT_VOICE = "fr-FR-VivienneMultilingualNeural"
                            
                            if waifu in WAIFU_VOICES:
                                waifu_specific_voices = WAIFU_VOICES[waifu]
                                UserData.voice = waifu_specific_voices.get(language, waifu_specific_voices.get("all", DEFAULT_VOICE))
                            else:
                                if not feautures["voices"]:
                                    feautures["voices"] = DataProcessing.generate_random_voices()
                                    waifu_dir = os.path.join(os.getcwd(), 'resources', waifu)
                                    system_file_path = os.path.join(waifu_dir, f'{waifu}.system.json')
                                    if os.path.exists(system_file_path):
                                        try:
                                            with open(system_file_path, 'r', encoding='utf-8') as f:
                                                system_data = json.load(f)
                                            if isinstance(system_data, dict) and "system" in system_data:
                                                system_data["voices"] = feautures["voices"]
                                            else:
                                                system_data = {"system": system_data, "voices": feautures["voices"]}
                                            
                                            with open(system_file_path, 'w', encoding='utf-8') as f:
                                                json.dump(system_data, f, indent=4, ensure_ascii=False)
                                        except Exception as e:
                                            print(f"Error updating system file with voices: {e}")
                                
                                UserData.voice = feautures["voices"].get(language, feautures["voices"].get("all", DEFAULT_VOICE))

                            UserData.expressions = [f["Name"] for f in feautures["expressions"]]
                            UserData.motions = {f["Name"]: i for i, f in enumerate(feautures["motions"])}
                            UserData.language, UserData.is_Live2D, UserData.usage = language, Live2D, usage
                            UserData.calls_since_last_sync = 0
                            self.last_sync_time = time()

                            if not use_microphone:
                                self.init_recorder()
                                self.listener_thread = Thread(target=self.listen, daemon=True)
                                self.listener_thread.start()

                            from Messages import Messages
                            self.Messages = Messages
                            self.response_thread = Thread(target=self.start_handle_responses, daemon=True)
                            self.response_thread.start()
                            
                            self.interface.gui.main_window.chat_window.set_speech_listener(self.audio_player)
                            self.interface.gui.main_window.chat_window.set_speech_listener(self)
                            return
                            
                        if self.interface.gui.stop_event.is_set():
                            print("[WAIT] Stop event detected, initiating shutdown...")
                            self.stop()
                            return
                    else:
                        print("[WAIT] GUI is None, interface was destroyed, initiating shutdown...")
                        self.stop()
                        return
                except Exception as e:
                    error(f"Error in wait_for_start loop: {e}")
                    self.stop()
                    return

        def start(self):
            try:
                print("[START] Initializing application...")
                # Start background loading of heavy modules
                Thread(target=self._load_heavy_modules_in_background, daemon=True).start()

                # Start the GUI thread immediately
                self.gui_thread = Thread(target=self.interface.start, args=(True,), daemon=False)
                self.gui_thread.start()

                self.wait_for_start()

                if not self.should_continue:
                    print("[START] Shutdown initiated during startup, exiting...")
                    return

                from Messages import Messages
                self.Messages = Messages
                self.Messages.user = self.name
                
                try:
                    Live2D.initializeAndRun(Messages.waifu)
                    sleep(1)
                    if self.interface and self.interface.gui and self.interface.gui.main_window:
                        self.interface.gui.main_window.chat_window.title_label.setText("YourOwnWaifu")
                except Exception as e:
                    print(f"Error initializing Live2D: {e}")
                    
                print("[START] Application initialization completed")
                
            except Exception as e:
                print(f"Critical error in start method: {e}")
                self.stop()

        def stop(self):
            if self._shutdown_started: return
            self._shutdown_started = True
            print("[SHUTDOWN] Application shutting down. Performing final usage sync...")
            
            self._force_shutdown_timer = Timer(15.0, self._force_shutdown)
            self._force_shutdown_timer.start()
            
            try:
                self.sync_usage_with_server()
                self.should_continue = False
                self.interrupt_event.set()
                
                if hasattr(self, 'Messages') and self.Messages:
                    self.Messages.stop = True
                    self.Messages.cleanup()
                
                components_to_stop = [
                    ("audio_player", lambda: self.audio_player.stop() if self.audio_player else None),
                    ("recorder", lambda: self.recorder.shutdown() if self.recorder else None),
                    ("waifu", lambda: self.waifu.kill() if self.waifu else None),
                    ("Live2D", lambda: Live2D.cleanup() if hasattr(Live2D, 'cleanup') else None)
                ]
                
                for name, stop_func in components_to_stop:
                    try: stop_func()
                    except Exception as e: print(f"Error stopping {name}: {e}")
                
                threads_to_join = [self.listener_thread, self.response_thread, self.gui_thread]
                if self.audio_player and hasattr(self.audio_player, 'streaming_thread'):
                    threads_to_join.append(self.audio_player.streaming_thread)
                
                for thread in threads_to_join:
                    if thread and thread.is_alive():
                        thread.join(timeout=3)
                
                if self.interface: self.interface.exit()
                
                if self._force_shutdown_timer: self._force_shutdown_timer.cancel()
                self._shutdown_complete.set()
                print("[SHUTDOWN] Graceful shutdown completed")
                
            except Exception as e:
                print(f"Error during shutdown: {e}")
                if self._force_shutdown_timer: self._force_shutdown_timer.cancel()
                self._force_shutdown()
            
            try: sys.exit(0)
            except: import os; os._exit(0)

        def _signal_handler(self, signum, frame):
            print(f"Received signal {signum}, initiating graceful shutdown...")
            self.stop()

        def _emergency_cleanup(self):
            if not self._shutdown_complete.is_set():
                print("Emergency cleanup triggered...")
                self.should_continue = False
                self.interrupt_event.set()

        def _force_shutdown(self):
            print("Force shutdown initiated - graceful shutdown took too long")
            import os; os._exit(1)

        def set_mute_state(self, is_muted):
            self.is_muted = is_muted
            print(f"Speech listener mute state set to: {is_muted}")

    def main():
        speech_listener = None
        try:
            speech_listener = SpeechListener()
            speech_listener.start()
            speech_listener.last_proactivity_check = time() + 250
            speech_listener.last_user_talk = time() + 400 
            
            while speech_listener.should_continue:
                sleep(0.5) 
                speech_listener.check_and_trigger_proactivity()
                if (speech_listener.gui_thread and not speech_listener.gui_thread.is_alive()) or \
                   (speech_listener.interface and hasattr(speech_listener.interface, 'is_destroyed') and speech_listener.interface.is_destroyed()):
                    print("[MAIN] GUI closed, initiating shutdown...")
                    break
                        
        except KeyboardInterrupt:
            print("Keyboard interrupt received, shutting down...")
        except Exception as e:
            print(f"Unexpected error in main: {e}")
        finally:
            if speech_listener and not speech_listener._shutdown_started:
                speech_listener.stop()

    main()