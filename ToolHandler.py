# tool_handler.py
from io import BytesIO
from PIL import ImageGrab
from google.genai import types
import Tools
from DataProcessing import UserData

class ToolHandler:
    def __init__(self):
        self.function_declarations = [
            {"name": "web_search", "description": "Search for the information if the user says about something that might be changed or you don't know", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query or question to look up."}}, "required": ["query"]}},
            {"name": "generate_image", "description": "Generate an image/photo", "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "Text description of the image/photo to generate and show"}}, "required": ["prompt"]}},
            {"name": "open_website", "description": "Open a website URL in the default browser", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The URL to open in the browser"}}, "required": ["url"]}},
            {"name": "check_screen", "description": "Take a screenshot and analyze what's currently on the screen", "parameters": {"type": "object", "properties": {"focus": {"type": "string", "description": "Optional focus area or element to pay attention to on the screen"}}, "required": []}},
            {"name": "change_app_volume", "description": "Change the volume of a specific application", "parameters": {"type": "object", "properties": {"app_name": {"type": "string", "description": "Name of the application to change volume for"}, "volume_change": {"type": "number", "description": "Volume change amount (-1.0 to 1.0, where -1.0 is mute and 1.0 is max volume)"}}, "required": ["app_name", "volume_change"]}}
        ]
        self.tools = types.Tool(function_declarations=self.function_declarations)

    def capture_screenshot(self) -> bytes | None:
        try:
            img = ImageGrab.grab()
            img = img.resize((img.width // 2, img.height // 2))
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG')
            return img_byte_arr.getvalue()
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return None

    async def execute_function_call(self, function_call) -> dict:
        function_name = function_call.name
        args = function_call.args
        try:
            if function_name == "generate_image":
                result = Tools.generate_image(args["prompt"], UserData.api)
                return {"result": result, "type": "text"}
            elif function_name == "open_website":
                result = Tools.open_website(args["url"])
                return {"result": result, "type": "text"}
            elif function_name == "check_screen":
                focus = args.get("focus", "")
                screenshot = self.capture_screenshot()
                return {"result": screenshot, "focus": focus, "type": "screenshot"}
            elif function_name == "change_app_volume":
                Tools.change_app_volume(args["app_name"], args["volume_change"])
                return {"result": f"Changed volume for {args['app_name']} by {args['volume_change']}", "type": "text"}
            elif function_name == "web_search":
                query = args.get("query", "")
                return {"result": f"Preparing to search the web for: {query}", "query": query, "type": "web_search"}
            else:
                return {"result": f"Unknown function: {function_name}", "type": "text"}
        except Exception as e:
            return {"result": f"Error executing {function_name}: {str(e)}", "type": "text"}