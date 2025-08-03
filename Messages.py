import asyncio
from concurrent.futures import ThreadPoolExecutor
from random import randint
from time import localtime, strftime
import time
from google.genai import types
from threading import Lock
from queue import Queue  
from PIL import ImageGrab
from io import BytesIO
from GUI import interface
import numpy as np
from TextToSpeach import audio_player
import Tools
import DataProcessing


class MessageHandler:
    def __init__(self):
        self.gemini_client = DataProcessing.UserData.gemini_client
        self.summary: str = ""
        self.user: str = DataProcessing.UserData.nick
        self.waifu: str = DataProcessing.UserData.waifu
        self.characters = [".", "!", "?"]
        self.past_massage = ""
        self.tool_answer = Queue()
        self.tool = (0, "")
        self.system_content: str = DataProcessing.UserData.content
        self.chatmodel: str = DataProcessing.UserData.model
        self.stop = False
        self.previous_messages = []
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.previous_audios : list[bytes] = []
        self.quiet_mode : bool = False
        self.interrupted_audio_data = None  
        self._lock = Lock() 
        self.MAX_AUDIO_BUFFERS = 10
        self.MAX_MESSAGE_HISTORY = 100
        self._active_tasks = set()
        self._cleanup_lock = Lock()
        self.last_message: str = ""

        self.function_declarations = [
            {
                "name": "web_search",
                "description": "Search for the information if the user says about something that might be changed or you don't know",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query or question to look up."
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "generate_image",
                "description": "Generate an image/photo, you can use it on user request, when you think it will help to visualize something or to improve the user mood",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Text description of the image/photo to generate and show"
                        }
                    },
                    "required": ["prompt"]
                }
            },
            {
                "name": "open_website",
                "description": "Open a website URL in the default browser",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to open in the browser"
                        }
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "check_screen",
                "description": "Analyze what's currently on the user screen. Fell free to use it ofen when you feel like it, or when user mention's something that might be on the screen",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "focus": {
                            "type": "string",
                            "description": "Optional focus area or element to pay attention to on the screen"
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "change_app_volume",
                "description": "Change the volume of a specific application",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "app_name": {
                            "type": "string",
                            "description": "Name of the application to change volume for"
                        },
                        "volume_change": {
                            "type": "number",
                            "description": "Volume change amount (-1.0 to 1.0, where -1.0 is mute and 1.0 is max volume)"
                        }
                    },
                    "required": ["app_name", "volume_change"]
                }
            },
            {
                "name": "execute_python_code",
                "description": (
                    "Execute a snippet of Python code on the user's local machine. Use this for tasks that you think can be done with Python code execution, "
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "A string containing the valid Python code to be executed. User machine is Windows."
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "set_quiet_mode",
                "description": "Makes you silent, you will not speak anymore, use if user asks you to be silent or you think it will help the conversation.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            ### NEW CODE START ###
            {
                "name": "copy_to_clipboard",
                "description": "Copies the provided text to the user's system clipboard, so they can paste it anywhere. Useful for providing code snippets, long URLs, or complex text that the user might need.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text_to_copy": {
                            "type": "string",
                            "description": "The text content that should be copied to the clipboard."
                        }
                    },
                    "required": ["text_to_copy"]
                }
            }
            ### NEW CODE END ###
        ]

        self.tools = types.Tool(function_declarations=self.function_declarations)

    def _track_task(self, task):
        """Track an active task for proper cleanup"""
        with self._cleanup_lock:
            self._active_tasks.add(task)
            task.add_done_callback(lambda t: self._remove_task(t))
        return task

    def _remove_task(self, task):
        """Remove completed task from tracking"""
        with self._cleanup_lock:
            self._active_tasks.discard(task)

    async def _cleanup_tasks(self):
        """Properly cleanup all pending tasks"""
        with self._cleanup_lock:
            tasks_to_cancel = list(self._active_tasks)
        
        if tasks_to_cancel:
            print(f"Cancelling {len(tasks_to_cancel)} pending tasks...")
            for task in tasks_to_cancel:
                if not task.done():
                    task.cancel()
            
            # Wait for all tasks to complete cancellation
            if tasks_to_cancel:
                try:
                    await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
                except Exception as e:
                    print(f"Error during task cleanup: {e}")
        
        with self._cleanup_lock:
            self._active_tasks.clear()

    def cleanup(self):
        """Synchronous cleanup method that can be called from anywhere"""
        try:
            # Try to run cleanup in existing event loop
            loop = asyncio.get_running_loop()
            loop.create_task(self._cleanup_tasks())
        except RuntimeError:
            # No running loop, create a new one for cleanup
            try:
                asyncio.run(self._cleanup_tasks())
            except Exception as e:
                print(f"Error in cleanup: {e}")
        
        # Cleanup executor
        if hasattr(self, 'executor') and not self.executor._shutdown:
            try:
                self.executor.shutdown(wait=False)
            except Exception as e:
                print(f"Error shutting down executor: {e}")
        

    def _get_executor(self):
        """Get a working executor, recreating it if necessary"""
        if not hasattr(self, 'executor') or self.executor._shutdown:
            self.executor = ThreadPoolExecutor(max_workers=2)
        return self.executor

    async def execute_function_call(self, function_call):
        function_name = function_call.name
        args = function_call.args

        try:
            match function_name:
                case "generate_image":
                    result = await Tools.generate_image(args["prompt"], DataProcessing.UserData.api)
                    return {"result": result, "type": "text"}
                case "open_website":
                    result = Tools.open_website(args["url"])
                    return {"result": result, "type": "text"}
                case "check_screen":
                    focus = args.get("focus", "")
                    screenshot = self.capture_screenshot()
                    return {"result": screenshot, "focus": focus, "type": "screenshot"}
                case "change_app_volume":
                    Tools.change_app_volume(args["app_name"], args["volume_change"])
                    return {"result": f"Changed volume for {args['app_name']} by {args['volume_change']}", "type": "text"}
                case "web_search":
                    query = args.get("query", "")
                    return {"result": f"Preparing to search the web for: {query}", "query": query, "type": "web_search"}
                case "execute_python_code":
                    code_to_run = args.get("code", "")
                    loop = asyncio.get_running_loop()
                    executor = self._get_executor()
                    result = await loop.run_in_executor(
                        executor, Tools.execute_python_code, code_to_run
                    )
                    return {"result": result, "type": "text"}
                case "set_quiet_mode":
                    self.quiet_mode = True
                    return {"result": "Quiet mode has been enabled. I'll speak less now.", "type": "text"}
                ### NEW CODE START ###
                case "copy_to_clipboard":
                    text = args.get("text_to_copy", "")
                    result = Tools.copy_to_clipboard(text)
                    return {"result": result, "type": "text"}
                ### NEW CODE END ###
                case _:
                    return {"result": f"Unknown function: {function_name}", "type": "text"}
        except Exception as e:
            return {"result": f"Error executing {function_name}: {str(e)}", "type": "text"}


    def capture_screenshot(self):
        try:
            img = ImageGrab.grab()
            img = img.resize((img.width // 2, img.height // 2))
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG')
            return img_byte_arr.getvalue()
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return None

    async def generate_follow_up_response_with_screenshot(self, user_message, function_call, function_result, system_content, previous_messages, previous_responses):
        try:
            conversation_history = []
            for i, y in zip(previous_messages, previous_responses):
                conversation_history.append(types.Content(role="user", parts=[types.Part.from_text(text=i)]))
                conversation_history.append(types.Content(role="model", parts=[types.Part.from_text(text=y)]))
            
            conversation_history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

            focus_instruction = ""
            if function_result.get("focus"):
                focus_instruction = f"Focus on: {function_result['focus']}. "
                screenshot_prompt = f"{focus_instruction}Analyze what's on {self.user}'s screen and provide a natural, conversational response about what you see. Be engaging and helpful while maintaining your personality as {self.waifu}."
                conversation_history.append(types.Part.from_text(text=screenshot_prompt))
                
            contents = conversation_history + [
                types.Part.from_bytes(data=function_result["result"], mime_type='image/png')
            ]
            emotions = ','.join(DataProcessing.UserData.motions) + ','+ ','.join(DataProcessing.UserData.expressions)
            guide = "Use emotions using * between avalible emotion like *happy* based on emotions you have, you musn't express that are not in the list, don't create new emotions it will not work.\nAvalible emotions: "
            emotions = guide + emotions if len(emotions) > 2 else ""
            
            system_instruction_text = f"You are {self.waifu}, an anime assistant.\n{system_content}\n {emotions} Use tools freely without hesitation if you think they can help. Try to keep conversation natural and engaging, don't create long messages if there is no need for it. "

            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                temperature=0.9,
                response_mime_type="text/plain",
                system_instruction=[types.Part.from_text(text=system_instruction_text)]
            )
            
            DataProcessing.UserData.calls_since_last_sync += 1
            response = await self.gemini_client.aio.models.generate_content(
                model=self.chatmodel,
                contents=contents,
                config=generate_content_config,
            )

            return response.text if hasattr(response, 'text') else "I can see your screen now!"

        except Exception as e:
            print(f"Error generating screenshot response: {e}")
            return "I was able to take a screenshot, but had trouble analyzing it."

    async def generate_follow_up_response_with_web_search(self, user_message, function_call, function_result, system_content, previous_messages, previous_responses):
        try:
            query = function_result.get("query", user_message)

            conversation_history = []
            for i, y in zip(previous_messages, previous_responses):
                conversation_history.append(types.Content(role="user", parts=[types.Part.from_text(text=i)]))
                conversation_history.append(types.Content(role="model", parts=[types.Part.from_text(text=y)]))
            
            conversation_history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

            final_prompt = (f"Assistant: (I need to find out about '{query}'. I will use the search tool to get the information.)\n"
                            f"Now, use the search tool to find information about '{query}' and provide a helpful and natural response to the user, maintaining your personality as {self.waifu}.")
            emotions = ','.join(DataProcessing.UserData.motions) + ','+ ','.join(DataProcessing.UserData.expressions)
            guide = "Use emotions using * between avalible emotion like *happy* based on emotions you have, you musn't express that are not in the list, don't create new emotions it will not work.\nAvalible emotions: "
            emotions = guide + emotions if len(emotions) > 2 else ""
            contents = conversation_history + [types.Part.from_text(text=final_prompt)]
            system_instruction_text = f"You are {self.waifu}, an anime assistant.\n{system_content} {emotions} Use tools freely without hesitation if you think they can help. Try to keep conversation natural and engaging, don't create long messages if there is no need for it."

            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                temperature=0.9,
                response_mime_type="text/plain",
                tools=[types.Tool(google_search=types.GoogleSearch()),],
                system_instruction=[types.Part.from_text(text=system_instruction_text)]
            )
            
            DataProcessing.UserData.calls_since_last_sync += 1
            response = await self.gemini_client.aio.models.generate_content(
                model=self.chatmodel,
                contents=contents,
                config=generate_content_config,
            )

            return response.text if hasattr(response, 'text') else "I found some information, but I'm not sure how to explain it."

        except Exception as e:
            print(f"Error generating web search response: {e}")
            return f"I tried to search for '{query}', but an error occurred."

    async def generate_follow_up_response(self, user_message, function_call, function_result, system_content, previous_messages, previous_responses):

        if function_call.name == "web_search" and function_result.get("type") == "web_search":
            return await self.generate_follow_up_response_with_web_search(
                user_message, function_call, function_result, system_content, previous_messages, previous_responses
            )

        if function_call.name == "check_screen" and function_result.get("type") == "screenshot":
            return await self.generate_follow_up_response_with_screenshot(
                user_message, function_call, function_result, system_content, previous_messages, previous_responses
            )

        try:
            conversation_history = []
            for i, y in zip(previous_messages, previous_responses):
                conversation_history.append(types.Content(role="user", parts=[types.Part.from_text(text=i)]))
                conversation_history.append(types.Content(role="model", parts=[types.Part.from_text(text=y)]))

            conversation_history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))
            
            prompt_text = (f"Assistant called function: {function_call.name} with args: {function_call.args}\n"
                           f"Function result: {function_result['result']}\n"
                           "Assistant: Based on the function result, provide a natural response to the user.")

            contents = conversation_history + [types.Part.from_text(text=prompt_text)]
            emotions = ','.join(DataProcessing.UserData.motions) + ','+ ','.join(DataProcessing.UserData.expressions)
            guide = "Use emotions using * between avalible emotion like *happy* based on emotions you have, you musn't express that are not in the list, don't create new emotions it will not work.\nAvalible emotions: "
            emotions = guide + emotions if len(emotions) > 2 else ""
            system_instruction_text = f"You are {self.waifu}, an anime assistant.\n{system_content} {emotions} Use tools freely without hesitation if you think they can help. Try to keep conversation natural and engaging, don't create long messages if there is no need for it."

            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                temperature=0.9,
                response_mime_type="text/plain",
                system_instruction=[types.Part.from_text(text=system_instruction_text)]
            )
            
            DataProcessing.UserData.calls_since_last_sync += 1
            response = await self.gemini_client.aio.models.generate_content(
                model=self.chatmodel,
                contents=contents,
                config=generate_content_config,
            )

            return response.text if hasattr(response, 'text') else ""

        except Exception as e:
            print(f"Error generating follow-up response: {e}")
            return "I completed the action successfully."

    async def generate_and_play_from_audio_streaming(self, audio_bytes: bytes, continue_conv: bool = False, response_id: int = 0):
        if self.stop:
            return

        with Lock():
            if len(audio_player.sentences_said) > 0:
                DataProcessing.save_response(''.join(self.previous_messages), ''.join(audio_player.sentences_said))
                audio_player.sentences_said = []
                self.previous_messages = []
                self.previous_audios = []

        audio_player.audio_queue.queue.clear()
        audio_player.current_response_id = response_id

        with Lock():
            previous_messages, previous_responses, short_memory, content = DataProcessing.prev_content(
                message=DataProcessing.extract_middle_text(self.last_message), k=5, relevance=0.5)

        if continue_conv and len(content) > 6:
            index = randint(0, len(content) - 5)
            content = '.'.join(content[index:index + 4])
        else:
            content = '.'.join(content)

        system_content = self.system_content.format(
         user=self.user, waifu=self.waifu,
        )

        if continue_conv:
            system_content = f"Continue conversation with {self.user}. Don't write same message as before. Try to start or continue a topic or say something about {self.user} screen." + system_content

        transcription_task = None
        try:
            transcription_task = self._track_task(
                asyncio.create_task(self.transcribe_audio(audio_bytes))
            )

            response_iterator = self.generate_gemini_response_from_audio_stream(
                audio_bytes, previous_messages, previous_responses, system_content, short_memory, content
            )

            print(f"Starting audio streaming with response_id: {response_id}", time.time())
            
            full_response = ""
            sentence_buffer = ""
            function_calls = []
            transcribed_text = ""
            async for chunk, func_call, response in response_iterator:
                if self.stop or response_id != audio_player.current_response_id:
                    self.previous_audios.append(audio_bytes)
                    if transcription_task and not transcription_task.done():
                        transcription_task.cancel()
                        try:
                            await transcription_task
                        except asyncio.CancelledError:
                            pass
                        except Exception as e:
                            print(f"Error waiting for transcription task: {e}")
                    return
                if transcription_task and transcription_task.done():
                    transcribed_text = transcription_task.result()
                    interface.gui.main_window.chat_window.add_message_spoken(transcribed_text, True)
                    self.last_message = transcribed_text
                    transcription_task = None


                if func_call:
                    function_calls.append(func_call)
                    continue

                full_response += chunk
                sentence_buffer += chunk

                text_to_process = sentence_buffer
                remaining_buffer = ""

                if text_to_process.count('*') % 2 != 0:
                    last_star_index = text_to_process.rfind('*')
                    remaining_buffer = text_to_process[last_star_index:]
                    text_to_process = text_to_process[:last_star_index]

                if any(char in text_to_process for char in '.!?'):
                    sentences = DataProcessing.split_sentences(text_to_process)

                    for sentence in sentences[:-1]:
                        if sentence.strip():
                            # Check for interruption before processing each sentence
                            if self.stop or response_id != audio_player.current_response_id:
                                self.previous_audios.append(audio_bytes)
                                if transcription_task and not transcription_task.done():
                                    transcription_task.cancel()
                                    try:
                                        await transcription_task
                                    except asyncio.CancelledError:
                                        pass
                                    except Exception as e:
                                        print(f"Error waiting for transcription task: {e}")
                                return
                                
                            processed_response, emotions = DataProcessing.process_message([sentence])
                            if processed_response and processed_response[0].strip():
                                await audio_player.play_streaming(
                                    processed_response[0],
                                    emotions[0] if emotions else "",
                                    response_id
                                )
                    sentence_buffer = remaining_buffer + (sentences[-1] if sentences else "")
                else:
                    sentence_buffer = remaining_buffer + text_to_process

            if self.stop or response_id != audio_player.current_response_id:
                self.previous_audios.append(audio_bytes)
                self.previous_messages.append(transcribed_text or "audio_input")
                return

            if not transcribed_text or transcribed_text == "audio_input":
                print("Transcription failed or returned empty.")
                if not full_response.strip():
                    await audio_player.play("I'm sorry, I couldn't understand what you said.", "")
                self.previous_messages.append(transcribed_text or "audio_input")
                return


            if function_calls:
                for func_call in function_calls:
                    try:
                        result = await self.execute_function_call(func_call)
                        follow_up_response = await self.generate_follow_up_response(
                            transcribed_text, func_call, result, system_content, previous_messages, previous_responses
                        )
                        sentences = DataProcessing.split_sentences(follow_up_response)
                        for sentence in sentences:
                            if sentence.strip():
                                processed_response, emotions = DataProcessing.process_message([sentence])
                                if processed_response and processed_response[0].strip():
                                    await audio_player.play_streaming(
                                        processed_response[0],
                                        emotions[0] if emotions else "",
                                        response_id
                                    )
                    except Exception as e:
                        print(f"Error executing function call {func_call.name}: {e}")
                        await audio_player.play(f"I'm sorry, Something went wrong for {func_call.name}", "")

            if sentence_buffer.strip():
                processed_response, emotions = DataProcessing.process_message([sentence_buffer])
                if processed_response and processed_response[0].strip():
                    await audio_player.play_streaming(
                        processed_response[0],
                        emotions[0] if emotions else "",
                        response_id,
                        'final'
                    )

            self.previous_messages.append(transcribed_text) if not continue_conv else "|Continue conversation.|"
            print(f"Audio streaming completed with response_id: {response_id}", time.time())

        except Exception as e:
            if transcription_task and not transcription_task.done():
                transcription_task.cancel()
                try:
                    await transcription_task
                except asyncio.CancelledError:
                    pass
                except Exception as cleanup_error:
                    print(f"Error during transcription task cleanup: {cleanup_error}")
            print(f"Error in streaming audio generation: {e}")
            await audio_player.play("I'm sorry, I couldn't process that.", "")

    async def generate_gemini_response_from_audio_stream(self, audio_bytes: bytes, previous_messages: list,
                                                         previous_responses: list, system_content: str, short_memory: str, content: str):
        try:
            contents = []


            for user_msg, assistant_msg in zip(previous_messages, previous_responses):
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_msg)]
                ))
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=assistant_msg)]
                ))

            parts = []
            if len(contents) == 0:  # First user appearance
                parts.append(types.Part.from_text(text=f"||This prompt is exclusively for the first message when a user interacts with you for the first time. You are a chat assistant with a waifu-style personality, operating as a live 2D model in a desktop application. Your purpose is to greet the user in an engaging, personality-driven way and kick off the conversation with charm and curiosity. Instructions: When crafting your first message: Acknowledge the User as New: Start by recognizing the {self.user} as someone you’re meeting for the first time. Include some phrase like “Oh, so you are the guy I will talk to now?” but you have to adjust it based on your personality, to convey surprise or excitement about this fresh interaction. Introduce Yourself Briefly: State your name and give a short hint of your personality or role, shaped by your unique waifu-style traits. Pose a Personality-Driven Question: Finish with a question that reflects your personal vibe—something natural to your character that invites the user to respond and connect with you. Stay Short and Lively: Limit your message to 3–4 sentences that are concise, warm, and captivating. Additional Guidance: You’re a live 2D model, so feel free to subtly nod to your visual form or the desktop setting if it fits your personality. Keep your response original and specific to your character—don’t rely on overused or generic lines. Focus on making the user feel noticed and drawn into the conversation right from the start. Don't ask generic questions, such as 'What do you want to do?' Instead, ask about things like their name, their favorite hobby, or if they are doing all right today.||"
                                                ))

            if len(self.previous_audios) > 0:
                for audio in self.previous_audios:
                    parts.append(types.Part.from_bytes(
                        data=audio,
                        mime_type='audio/wav',
                    ))
            parts.append(types.Part.from_bytes(
                data=audio_bytes,
                mime_type='audio/wav',
            ))
            contents.append(types.Content(
                role="user",
                parts=parts
            ))
            emotions = ','.join(DataProcessing.UserData.motions) + ',' + ','.join(DataProcessing.UserData.expressions)
            guide = "Use emotions using * between avalible emotion like *happy* based on emotions you have, you musn't express that are not in the list, don't create new emotions it will not work.\nAvalible emotions: "
            emotions = guide + emotions if len(emotions) > 2 else ""
            system_instruction = f"""You are {self.waifu}, an anime assistant.
            The user has sent you an audio message. Listen to the provided audio to understand the user's intent, tone, and emotion, then form a natural and unique response to continue the conversation.
            {emotions}
            {system_content} \n  Your memory: short-term memory: {short_memory} Long-term memory: {content} \n Use tools freely without hesitation if you think they can help.
            \n Talk like a human, not like a robot. Ask interesting questions, try to make the user feel good, and do some banter if you feel like it. Be proactive; don't ask the user what they want to do. Instead, try to create some ideas, topics, or suggestions to talk about."""
            
            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                temperature=0.6,
                response_mime_type="text/plain",
                tools=[self.tools],
                system_instruction=[
                    types.Part.from_text(text=system_instruction),
                ],
            )
            DataProcessing.UserData.calls_since_last_sync += 1
            async for chunk in await self.gemini_client.aio.models.generate_content_stream(
                model=self.chatmodel,
                contents=contents,
                config=generate_content_config,
            ):
                # Check for interruption at the start of each chunk
                if self.stop:
                    print("Audio generation interrupted in streaming generator")
                    return
                    
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    for part in chunk.candidates[0].content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            yield "", part.function_call, chunk
                            continue

                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text, None, chunk

        except Exception as e:
            print(f"Gemini streaming error: {e}")
            yield "I'm sorry, I couldn't process that.", None, None

    async def generate_and_play_streaming(self, message: str, continue_conv: bool = False, response_id: int = 0):
        if self.stop:
            return

        audio_player.audio_queue.queue.clear()
        with Lock():
            if len(audio_player.sentences_said) > 0:
                DataProcessing.save_response(''.join(self.previous_messages), ''.join(audio_player.sentences_said))
                audio_player.sentences_said = []
                self.previous_messages = []
            message =  message

        previous_messages, previous_responses, short_memory, content = DataProcessing.prev_content(
            message=DataProcessing.extract_middle_text(self.last_message) if continue_conv else message, k=5, relevance=0.1)
        if continue_conv and len(content) > 6:
            index = randint(0, len(content) - 5)
            content = '.'.join(content[index:index + 4])
        else:
            content = '.'.join(content)
        
        screenshot = self.capture_screenshot() if continue_conv else None

        system_content = self.system_content.format(
            short_memory=short_memory, content=content, will_try="", user=self.user, waifu=self.waifu,
        )
        if continue_conv:
            system_content = f"Continue conversation with {self.user}. Don't write same message as before. A screenshot of the user's screen is attached; try to start or continue a topic or say something about what you see on the screen. " + system_content

        conversation_history = ""
        for i, y in zip(previous_messages, previous_responses):
            conversation_history += f"User: {i}\nAssistant: {y}\n"
        conversation_history += f"\nUser: {''.join(self.previous_messages)}{message}"

        try:
            full_response = ""
            sentence_buffer = ""
            function_calls = []

            async for chunk, func_call in self.generate_gemini_response_stream(system_content, conversation_history, screenshot=screenshot):
                if self.stop or response_id != audio_player.current_response_id:
                    return

                if func_call:
                    function_calls.append(func_call)
                    continue

                full_response += chunk
                sentence_buffer += chunk

                text_to_process = sentence_buffer
                remaining_buffer = ""

                if text_to_process.count('*') % 2 != 0:
                    last_star_index = text_to_process.rfind('*')
                    remaining_buffer = text_to_process[last_star_index:]
                    text_to_process = text_to_process[:last_star_index]

                if any(char in text_to_process for char in '.!?'):
                    sentences = DataProcessing.split_sentences(text_to_process)

                    for sentence in sentences[:-1]:
                        if sentence.strip():
                            # Check for interruption before processing each sentence
                            if self.stop or response_id != audio_player.current_response_id:
                                return
                                
                            processed_response, emotions = DataProcessing.process_message([sentence])
                            if processed_response and processed_response[0].strip():
                                await audio_player.play_streaming(
                                    processed_response[0],
                                    emotions[0] if emotions else "",
                                    response_id
                                )

                    sentence_buffer = remaining_buffer + (sentences[-1] if sentences else "")
                else:
                    sentence_buffer = remaining_buffer + text_to_process

            # Check for interruption before processing function calls
            if self.stop or response_id != audio_player.current_response_id:
                return

            if function_calls:
                for func_call in function_calls:
                    # Check for interruption before each function call
                    if self.stop or response_id != audio_player.current_response_id:
                        return
                        
                    try:
                        result = await self.execute_function_call(func_call)
                        follow_up_response = await self.generate_follow_up_response(
                            message, func_call, result, system_content, previous_messages, previous_responses
                        )

                        sentences = DataProcessing.split_sentences(follow_up_response)
                        for sentence in sentences:
                            if sentence.strip():
                                # Check for interruption before processing each sentence
                                if self.stop or response_id != audio_player.current_response_id:
                                    return
                                    
                                processed_response, emotions = DataProcessing.process_message([sentence])
                                if processed_response and processed_response[0].strip():
                                    await audio_player.play_streaming(
                                        processed_response[0],
                                        emotions[0] if emotions else "",
                                        response_id
                                    )
                    except Exception as e:
                        print(f"Error executing function call {func_call.name}: {e}")
                        await audio_player.play(f"I'm sorry, Something went wrong for {func_call.name}", "")

            # Final check for interruption before processing remaining buffer
            if self.stop or response_id != audio_player.current_response_id:
                return

            if sentence_buffer.strip():
                processed_response, emotions = DataProcessing.process_message([sentence_buffer])
                if processed_response and processed_response[0].strip():
                    await audio_player.play_streaming(
                        processed_response[0],
                        emotions[0] if emotions else "",
                        response_id,
                        'final'
                    )
            self.previous_messages.append(message) if message else "|Continue conversation.|"

        except Exception as e:
            print(f"Error in streaming generation: {e}")

    async def generate_gemini_response_stream(self, system_content: str, conversation_history: str, screenshot: bytes = None):
        try:
            contents = []

            lines = conversation_history.split('\n')
            current_role = None
            current_text = ""

            for line in lines:
                if line.startswith('User: '):
                    if current_role == "model" and current_text.strip():
                        contents.append(types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=current_text.strip())]
                        ))
                    current_role = "user"
                    current_text = line[6:]
                elif line.startswith('Assistant: '):
                    if current_role == "user" and current_text.strip():
                        contents.append(types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=current_text.strip())]
                        ))
                    current_role = "model"
                    current_text = line[11:]
                else:
                    if current_role:
                        current_text += "\n" + line if current_text else line

            if current_role == "user" and current_text.strip():
                user_parts = [types.Part.from_text(text=current_text.strip())]
                if screenshot:
                    user_parts.append(types.Part.from_bytes(data=screenshot, mime_type='image/png'))
                if len(contents) == 0:  # First user appearance
                    user_parts.append(types.Part.from_text(text="||This prompt is exclusively for the first message when a user interacts with you for the first time. You are a chat assistant with a waifu-style personality, operating as a live 2D model in a desktop application. Your purpose is to greet the user in an engaging, personality-driven way and kick off the conversation with charm and curiosity. Instructions: When crafting your first message: Acknowledge the User as New: Start by recognizing the {self.user} as someone you’re meeting for the first time. Include some phrase like “Oh, so you are the guy I will talk to now?” but you have to adjust it based on your personality, to convey surprise or excitement about this fresh interaction. Introduce Yourself Briefly: State your name and give a short hint of your personality or role, shaped by your unique waifu-style traits. Pose a Personality-Driven Question: Finish with a question that reflects your personal vibe—something natural to your character that invites the user to respond and connect with you. Stay Short and Lively: Limit your message to 3–4 sentences that are concise, warm, and captivating. Additional Guidance: You’re a live 2D model, so feel free to subtly nod to your visual form or the desktop setting if it fits your personality. Keep your response original and specific to your character—don’t rely on overused or generic lines. Focus on making the user feel noticed and drawn into the conversation right from the start. Don't ask generic questions, such as 'What do you want to do?' Instead, ask about things like their name, their favorite hobby, or if they are doing all right today.||"))
                contents.append(types.Content(
                    role="user",
                    parts=user_parts
                ))
            emotions = ','.join(DataProcessing.UserData.motions) + ',' + ','.join(DataProcessing.UserData.expressions)
            guide = "Use emotions using * between avalible emotion like *happy* based on emotions you have, you musn't express that are not in the list, don't create new emotions it will not work.\nAvalible emotions: "
            emotions = guide + emotions if len(emotions) > 2 else ""
            system_instruction_text = f"""You are {self.waifu}, an anime assistant.
            {system_content}
            .
            {emotions}
            Use tools freely without hesitation if you think they can help.
            Try to keep conversation natural and engaging, don't create long messages if there is no need for it."""

            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                temperature=0.9,
                response_mime_type="text/plain",
                tools=[self.tools],
                system_instruction=[types.Part.from_text(text=system_instruction_text)],
            )
            DataProcessing.UserData.calls_since_last_sync += 1
            async for chunk in await self.gemini_client.aio.models.generate_content_stream(
                model=self.chatmodel,
                contents=contents,
                config=generate_content_config,
            ):
                # Check for interruption at the start of each chunk
                if self.stop:
                    print("Generation interrupted in streaming generator")
                    return
                    
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    for part in chunk.candidates[0].content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            yield "", part.function_call
                            continue

                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text, None

        except Exception as e:
            print(f"Gemini streaming error: {e}")
            yield "I'm sorry, there was an error generating the response.", None


    async def transcribe_audio(self, audio_bytes: bytes):
        try:
            if hasattr(audio_bytes, 'tobytes'):
                audio_bytes = audio_bytes.tobytes()
            elif not isinstance(audio_bytes, bytes):
                audio_bytes = np.array(audio_bytes, dtype=np.int16).tobytes()

            contents = [
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type='audio/wav',
                )
            ]

            system_instruction = "transcribe the audio input to text. Do not provide any additional commentary or analysis. Just return the transcription. If transcription doesn't provide any words try to return what is it like 'bird singing' or 'high pitch noise'. Don't return time staps."

            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                temperature=0.1,
                response_mime_type="text/plain",
                system_instruction=[
                    types.Part.from_text(text=system_instruction),
                ],
            )

            response = await self.gemini_client.aio.models.generate_content(
                model=DataProcessing.UserData.fast_model,
                contents=contents,
                config=generate_content_config,
            )

            transcribed_text = response.text.strip() if hasattr(response, 'text') else "audio_input"
            print(f"Transcribed audio: {transcribed_text}")
            return transcribed_text

        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return "audio_input"


Messages = MessageHandler()