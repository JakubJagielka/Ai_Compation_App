import contextlib
import io
import os
from PIL import Image
from uuid import uuid4
from webbrowser import WindowsDefault, open as webopen
from io import BytesIO
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
from google import genai
import pyperclip

async def generate_image(prompt: str, api) -> str:
    """Use the tool to generate an image with Gemini."""
    try:
        api_key = api
        if not api_key:
            return "Failed to generate an image: GEMINI_API_KEY environment variable not set."

        client = genai.Client(api_key=api_key)

        response = await client.aio.models.generate_images(
            model="models/imagen-4.0-generate-preview-06-06",
            prompt=prompt,
            config=dict(
                number_of_images=1,
                output_mime_type="image/jpeg",
                person_generation="ALLOW_ADULT",
                aspect_ratio="1:1",
            ),
        )

        if not response.generated_images:
            return f"Failed to generate an image for: {prompt}. No images returned."

        generated_image_data = response.generated_images[0].image.image_bytes # Accessing bytes directly

        if not os.path.exists("images"):
            os.mkdir("images")
        
        path = f"images/img_{uuid4()}.png" 

        image = Image.open(BytesIO(generated_image_data))
        
        image.save(path)
        image.show()
        
        return f"Successfully generated and shown an image of: {prompt}."
    except AttributeError as e:
        return f"Failed to generate an image due to API response structure: {str(e)}"
    except Exception as e:
        return f"Failed to generate an image: {str(e)}"


def open_website( url: str) -> str:
    """Use the tool."""
    url = url.strip()
    try:
        WindowsDefault()
        webopen(url)
        return f"Successfully opened {url} in the browser."
    except Exception as e:
        return f"Failed to open {url}. Error: {str(e)}"

def change_app_volume(app_name, volume_change):
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        volume = session._ctl.QueryInterface(ISimpleAudioVolume)
        if session.Process and session.Process.name() == app_name:
            current_volume = volume.GetMasterVolume()
            new_volume = max(0.0, min(1.0, current_volume + volume_change))
            volume.SetMasterVolume(new_volume, None)
            return

def copy_to_clipboard(text_to_copy: str) -> str:
    """
    Copies the given text to the system's clipboard.

    Args:
        text_to_copy: The string of text to be copied.

    Returns:
        A confirmation message indicating success or failure.
    """
    try:
        pyperclip.copy(text_to_copy)
        print(f"Successfully copied to clipboard: '{text_to_copy[:50]}...'")
        return f"The text has been successfully copied to the clipboard."
    except pyperclip.PyperclipException as e:
        error_message = (f"Failed to copy to clipboard. Error: {e}. "
                         "On Linux, please ensure 'xclip' or 'xsel' is installed.")
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"An unexpected error occurred while copying to clipboard: {e}"
        print(error_message)
        return error_message
    
    
def execute_python_code(code: str) -> str:
    """
    Executes a string of Python code on the local machine and returns its output or any errors.
    
    """

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        # Create a clean, but not sandboxed, local scope for the execution
        local_scope = {}
        
        # Use contextlib to redirect stdout and stderr to our StringIO objects
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(code, globals(), local_scope)

        # Get the captured output and errors
        stdout_result = stdout_capture.getvalue()
        stderr_result = stderr_capture.getvalue()

        # Format the result string to be returned to the LLM
        result = ""
        if stdout_result:
            result += f"Output:\n{stdout_result}\n"
        if stderr_result:
            result += f"Errors:\n{stderr_result}\n"

        # If there was no output or error, return a success message
        if not result:
            return "Code executed successfully with no output."
        
        return result.strip()

    except Exception as e:
        # Catch any exceptions that occurred during the compilation or execution itself
        return f"An exception occurred while trying to execute the code: {type(e).__name__}: {str(e)}"