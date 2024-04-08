from langchain_community.llms import Ollama

class OllamaMistral:
    
    @staticmethod
    def get_llm(url):
        return Ollama(base_url=url, model="mistral")