import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_llm(provider: str, model_name: str, temperature: float = 0):
    provider = provider.lower().strip()
    if provider == "ollama":
        print(f"   🔌 Loading Local Model: {model_name}")
        return ChatOllama(model=model_name, temperature=temperature)
    elif provider == "google":
        print(f"   ☁️ Loading Cloud Model: {model_name}")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY not found in .env file")
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=api_key)
    else:
        raise ValueError(f"❌ Unknown provider: {provider}")
