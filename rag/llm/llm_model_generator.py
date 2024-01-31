from transformers import (
  AutoTokenizer, 
  AutoModelForCausalLM, 
  BitsAndBytesConfig,
  pipeline
)
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain
from rag.config import Config

class LLMModelGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.model_name = self.config.llm.model_name
        self.bnb_config = self._get_bits_and_bytes_config()
        self.model = self._get_model(self.bnb_config)
        self.tokenizer = self._get_tokenizer()
        self.text_generation_pipeline = self._get_text_generation_pipeline(self.model, self.tokenizer)
    
    def get_llm_model(self):
        return HuggingFacePipeline(pipeline=self.text_generation_pipeline)
    
    def _get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer
    
    def _get_bits_and_bytes_config(self):
        bnb_config = BitsAndBytesConfig(
            **self.config.bits_and_bytes
        )
        return bnb_config
    
    def _get_model(self, bnb_config):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
        )
        return model
    
    def print_number_of_trainable_model_parameters(self):
        trainable_model_params = 0
        all_model_params = 0
        for _, param in self.model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        return print(f"trainable model parameters: {trainable_model_params}\n"\
                    f"all model parameters: {all_model_params}\n"\
                    f"percentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%")
    
    def _get_text_generation_pipeline(self, model, tokenizer):
        # pipeline params in below file
        # /home/atul/.local/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:40
        text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            **self.config.llm.params
        )
        return text_generation_pipeline