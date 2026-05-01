import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

try:
    for model in client.models.list():
        if "generateContent" in model.supported_actions:
            print(model.name)
except Exception as e:
    print("Error listing models:", e)
