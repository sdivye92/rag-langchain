from typing import Union
from transformers import (
  AutoTokenizer,
  PreTrainedModel,
  TFPreTrainedModel,
  pipeline
)
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain
from rag.config import Config

class PipelineGenerator:
    @classmethod
    def get_pipeline(cls, config: Config,
                     model: Union[PreTrainedModel, TFPreTrainedModel]):
        tokenizer = cls._get_tokenizer(model)
        text_generation_pipeline = cls._get_text_generation_pipeline(model, tokenizer, config.llm.params)
        return HuggingFacePipeline(pipeline=text_generation_pipeline)
    
    @staticmethod
    def _get_tokenizer(model: Union[PreTrainedModel, TFPreTrainedModel]):
        tokenizer = AutoTokenizer.from_pretrained(model.name_or_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer
    
    @staticmethod
    def _get_text_generation_pipeline(model: Union[PreTrainedModel, TFPreTrainedModel],
                                      tokenizer, llm_params):
        # pipeline params in below file
        # /home/atul/.local/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:40
        text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            **llm_params
        )
        return text_generation_pipeline