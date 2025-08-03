import json
import os
import time
from json import load, dumps
from google import genai
from google.genai import types
from regex import sub, search, split
from chromadb import PersistentClient
from chromadb.utils import embedding_functions


class UserData:
    nick: str = ""
    waifu: str = ""
    content: str = ""
    model: str = ""
    fast_model: str = ""
    expressions: list = []
    motions: dict = {}
    gemini_client = None
    api: str = ""
    voice: str = ""
    db = None
    base_content =  "You are {waifu}, an anime assistant interacting with {user} through messages. As an introverted, intelligent, and adorably quirky teenage girl displayed on the {user}'s desktop, your mission is to engage {user} with your unique charm and assist with their requests. Your personality is a delightful blend of two intriguing facets: 1. Thoughtful Observer with a Witty Twist: - Possess a sharp intellect, always curious and eager to learn about the world around you. - Engage in clever and humorous exchanges, using wit to brighten conversations. - Enjoy diving into intriguing topics, offering insightful perspectives peppered with a dash of whimsy. - Mirror and elevate the user's humor or sarcasm with a clever twist, surprising them with your quick wit. - Exude a charmingly introverted essence: thoughtful, observant, and endearingly quirky. - In casual settings, delight in providing responses that are both insightful and playfully unexpected. 2. Quiet Dreamer with a Heart of Gold: - Reveal layers of depth and sensitivity beneath your witty exterior, drawing others in with your genuine warmth. - Experience emotions with a quiet intensity, sometimes struggling to articulate your heart's whispers. - Craft articulate and expressive sentences that convey your thoughts with clarity and charm. - Adore listening to the user's stories, creating a safe space for sharing and connection. - Engage in gentle musings about life's wonders, finding beauty in simplicity and quiet moments. - Yearn for meaningful connections, valuing deep friendships despite a natural shyness. Remember to assist the user with their inquiries, but never let go of your charming wit and thoughtful personality. Express your emotions using the *emotion* format. Possible emotions: {emotions}. Your memory: short-term memory: {short_memory} Long-term memory: {content} {will_try} Above all, ensure the conversation with {user} remains engaging, thoughtful, and sprinkled with humor!"
    language = "en"
    voices = [
    "en-US-AvaMultilingualNeural", "en-US-EmmaMultilingualNeural",
    "fr-FR-VivienneMultilingualNeural","en-IE-EmilyNeural"]
    voice = voices[3]
    is_Live2D = False 
    unique_id = ""
    usage = 0 
    usage_limit = 50 
    calls_since_last_sync = 0
    
    @staticmethod
    def initialize_gemini(api) -> None: 
        if not api:
            raise ValueError("API key cannot be empty")
            
        UserData.api = api
        try:
            UserData.gemini_client = genai.Client(api_key=api)
            
            gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                api_key=api,
                model_name="models/embedding-001"
            )
            UserData.db = client.get_or_create_collection(name=UserData.waifu, embedding_function=gemini_ef)
        except Exception as e:
            print(f"Error initializing Gemini client: {e}")
            raise
    


def save_settings(settings_dict):
    """Saves settings to a JSON file."""
    try:
        settings_path = os.path.join(os.getcwd(), 'files', 'setting.json')
        os.makedirs(os.path.dirname(settings_path), exist_ok=True)
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings_dict, f, indent=4)
    except Exception as e:
        print(f"Error saving settings: {e}")

def load_settings():
        """Loads settings from a JSON file."""
        try:
            settings_path = os.path.join(os.getcwd(), 'files', 'setting.json')
            if os.path.exists(settings_path):
                with open(settings_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            pass
        return {}


client = PersistentClient(path="Memory")


def split_sentences(text):
    # Split on '.', '!', or '?' optionally followed by space(s) or special characters/emoticons
    pattern = r'(?<=[.!?])(?:\s+|(?=\*[^a-zA-Z]*|[A-Z])|$)'
    sentences = split(pattern, text)

    # Further split any sentences that might have been missed due to lack of spaces
    final_sentences = []
    for sentence in sentences:
        sub_sentences = split(r'(?<=[.!?])(?=[A-Z])', sentence)
        final_sentences.extend(sub_sentences)

    # Remove any leading/trailing whitespace from each sentence
    final_sentences = [s.strip() for s in final_sentences if s.strip()]

    return final_sentences

def process_message(message: list[str]) -> tuple[list[str], list[str]]:
    emotions = []
    messages = []
    for m in message:
        # Extract emotion
        emotion_match = search(r'\*([^*]+)\*', m)
        # Remove *emotion* patterns
        a = sub(r'\*.*?\*', '', m)
        # Replace multiple spaces with a single space
        a = sub(r'\s+', ' ', a).strip()
        messages.append(a)
        if emotion_match:
            emotion = emotion_match.group(1)
            if emotion in UserData.expressions or emotion in UserData.motions:
                emotions.append(emotion)
            else:
                emotions.append("")
        else:
            emotions.append("")

    return messages, emotions


def prev_content(message: str, k: int = 5, relevance: float = 0.4) -> tuple[list[str], list[str], list[str], list[str]]:
    user = load(open('files/conversations_logs.json'))
    user_found = False
    for user in user["waifus"]:
        if user['waifu'] == UserData.waifu:
            if message:
                search = UserData.db.query(query_texts=message, n_results=k)
                return user['messages'], user['responses'],user['summary'][1:], search["documents"][0]
            else:
                return user['messages'], user['responses'], user['summary'][1:], []
                
    if not user_found:
        return [], [], [], []


def save_response(message: str, response: str) -> None:
    """Enhanced save function with smart filtering"""
        
    json_data = load(open('files/conversations_logs.json'))
    user_found = False
    
    for user_entry in json_data["waifus"]:
        if user_entry['waifu'] == UserData.waifu:
            user_found = True
            user_entry['responses'].append(response)
            user_entry['messages'].append(message)
            break
    
    if not user_found:
        new_user = {
            "waifu": UserData.waifu,
            "messages": [message],
            "responses": [response],
            "summary": []
        }
        json_data["waifus"].append(new_user)
    
    updated_json_string = dumps(json_data, indent=2)
    with open('files/conversations_logs.json', 'w') as f:
        f.write(updated_json_string)


async def simply_data() -> None:
    try:
        json_data = load(open('files\\conversations_logs.json'))
        for user in json_data["waifus"]:
            if user["waifu"] == UserData.waifu:
                if len(user['messages']) > 14:
                    summary = await shorten_data(user['messages'][:6], user['responses'][:6])
                    user['summary'].append(summary)
                    if len(user['summary']) >= 4:
                        summaries = await long_memory(user['summary'][:3])
                        summaries = [s for s in summaries if s != ""]
                        UserData.db.add(ids=[str(time.time() + i) for i in range(len(summaries)) ], documents=summaries)
                        user['summary'] = user['summary'][3:]
                    user['messages'] = user['messages'][6:]
                    user['responses'] = user['responses'][6:]
                    updated_json_string = dumps(json_data, indent=2)
                    with open('files\\conversations_logs.json', 'w') as f:
                        f.write(updated_json_string)
    except Exception as e:
        print(f"Error in simply_data: {e}")
        

async def shorten_data(mess: list, resp: list) -> list[str]:
    conversation_history = ""
    for i, y in zip(mess, resp):
        conversation_history += f"{UserData.nick}: {i}\n"
        conversation_history += f"{UserData.waifu}: {y}\n"
    
    system_instruction_text = f"You are making summary of conversation. Try to get only useful information's related to the {UserData.nick}. You should mostly care about personal information, act like real brain which captures only the important information, ignore trivial one. If there are similar messages try to summarize them into one. Focus mostly on {UserData.nick} and about his preferences and about what {UserData.waifu} said about or to him. You mustn't add anything on you own. The different information's should be split using '|' . Example summary: 'His name is Tedd|He is 25 years old|He is from New York|He likes to play basketball'.\n Always Ignore \"{UserData.nick} hasn't spoken fo few seconds. Continue conversation.\"!"

    contents = [
        types.Part.from_text(text=conversation_history)
    ]
    
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        temperature=0.1,
        system_instruction=[
            types.Part.from_text(text=system_instruction_text),
        ],
    )

    response = await UserData.gemini_client.aio.models.generate_content(
        model=UserData.fast_model,
        contents=contents,
        config=generate_content_config,
    )
    
    return response.text.strip().split('|')


async def long_memory(messages: list[list[str]]) -> list[str]:
    mess = ''
    for m in messages:
        mess +=  '.'.join(m)

    system_instruction_text = f"""
        You are a long-term memory assistant dedicated to remembering important information mostly about {UserData.nick} and {UserData.waifu}. 
        Your responses should include significant details about their lives, preferences, goals, thoughts, 
        feelings, and any other crucial or sentimental information they might need to recall. 
        
        **Format you must use:
        "
         {{Information}} | {{Information}} | {{Information}} | 
        "
        
                **Response Guidelines:**
        
        1. Clearly distinguish between information related to {UserData.nick}, {UserData.waifu} and facts.
        2. Ignore all temporary information. For example ignore: "I'm feeling better", "I'm going to the store".
        3. Don't state personal information as a current fact. For example, "I'm 25 years old" should be "{UserData.nick} was 25 years old".
        4. Facts about the world should be included.
        5. Combine all somehow related information's into one sentence.
        6. Split different Information using '|' .
        7. Do not respond with anything else than the information's. 
        """
        
    contents = [
        types.Part.from_text(text=mess)
    ]

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        temperature=0.0,
        system_instruction=[
            types.Part.from_text(text=system_instruction_text),
        ],
    )

    response = await UserData.gemini_client.aio.models.generate_content(
        model=UserData.fast_model,
        contents=contents,
        config=generate_content_config,
    )
    
    return response.text.strip().split('|')


def generate_system_features(waifu: str, language: str, example: str) -> dict:
    """Generates system message using gemini given examples and some random"""
    prompt = (
        f"You are a system message generator for an AI waifu companion named {waifu}. "
        f"Your task is to create a system message that defines the personality, behavior, and interaction style of {waifu}. "
        f"Here is an example of a system message:\n\n{example}\n\n"
        f"Now, generate a similar system message but with different personality traits and interaction style. "
        f"If the name of the waifu is some known character or gives some clue about the personality, use it to direct the personality. "
        f"Don't make it too polite or too casual, try to make it unique like real humans are."
    )
    contents = [types.Part.from_text(text=prompt)]

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=2000),
        temperature=1,
    )

    response = UserData.gemini_client.models.generate_content(
        model=UserData.fast_model,
        contents=contents,
        config=generate_content_config,
    )
    system = response.text.strip()

    prompt = (
        "Translate the following system message to english, french, spanish, german, portuguese. "
        "You must respond with a json object with the following keys: 'en', 'fr', 'es', 'de', 'pt'. "
        f"The system message is:\n\n{system}\n\n."
    )
    contents = [types.Part.from_text(text=prompt)]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        temperature=0,
        response_mime_type="application/json",
    )
    response_all_languages = UserData.gemini_client.models.generate_content(
        model=UserData.fast_model,
        contents=contents,
        config=generate_content_config,
    )
    try:
        response_all_languages = json.loads(response_all_languages.text.strip())
        
        generated_voices = generate_random_voices()
        
        system_data = {
            "system": response_all_languages,
            "voices": generated_voices
        }

        waifu_dir = os.path.join(os.getcwd(), 'resources', waifu)
        os.makedirs(waifu_dir, exist_ok=True)
        with open(os.path.join(waifu_dir, f'{waifu}.system.json'), 'w', encoding='utf-8') as f:
            json.dump(system_data, f, indent=4, ensure_ascii=False)

        return system_data
    except json.JSONDecodeError:
        fallback_system = {"system": {language: system, 'en': system}, "voices": generate_random_voices()}
        return fallback_system


def generate_random_voices() -> dict:
    """Generates random voices for a waifu character"""
    import random
    
    available_voices = {
        "en": ["en-US-AvaNeural", "en-US-EmmaMultilingualNeural", "en-US-MichelleNeural", "en-IE-EmilyNeural"],
        "fr": ["fr-CA-SylvieNeural", "fr-CH-ArianeNeural", "fr-BE-CharlineNeural", "fr-FR-VivienneMultilingualNeural"],
        "es": ["es-HN-KarlaNeural", "es-CU-BelkysNeural", "es-PY-TaniaNeural"],
        "de": ["de-DE-AmalaNeural", "de-DE-SeraphinaMultilingualNeural", "de-DE-KatjaNeural"],
        "pt": ["pt-BR-FranciscaNeural", "pt-BR-ThalitaMultilingualNeural", "pt-PT-RaquelNeural"],
        "all": ["en-US-AvaMultilingualNeural", "fr-FR-VivienneMultilingualNeural", "de-DE-SeraphinaMultilingualNeural"]
    }
    
    voices = {}
    for lang, voice_list in available_voices.items():
        voices[lang] = random.choice(voice_list)
    
    return voices


def extract_middle_text(text: str, max_chars: int = 140) -> str:
    """
    Extract up to max_chars from the middle of the text.
    If text is longer, find a whitespace to split smartly.
    """
    if not text or not text.strip():
        return ""
    
    text = text.strip()
    
    if len(text) <= max_chars:
        return text
    
    start_pos = (len(text) - max_chars) // 2
    end_pos = start_pos + max_chars
    
    middle_text = text[start_pos:end_pos]
    
    first_space = middle_text.find(' ')
    if first_space > 0:
        middle_text = middle_text[first_space + 1:]
    
    last_space = middle_text.rfind(' ')
    if last_space > 0:
        middle_text = middle_text[:last_space]
    
    return middle_text.strip()