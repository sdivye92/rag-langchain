from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

class Gemini:
    
    @classmethod
    def get_llm(cls, google_api_key, model="gemini-pro", **kwargs):
        return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=google_api_key,
                **kwargs
            )