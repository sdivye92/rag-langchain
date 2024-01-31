from typing import Union
from transformers import (
  AutoModelForCausalLM, 
  PreTrainedModel,
  TFPreTrainedModel,
  BitsAndBytesConfig
)
from rag.config import Config

Model = Union[PreTrainedModel, TFPreTrainedModel]

class LLMModelGenerator:
    @classmethod
    def get_model(cls, config: Config) -> Union[PreTrainedModel, TFPreTrainedModel]:
        bnb_config = cls._get_bits_and_bytes_config(config.bits_and_bytes)
        model = AutoModelForCausalLM.from_pretrained(
            config.llm.model_name,
            quantization_config=bnb_config,
        )
        return model

    @classmethod
    def print_number_of_trainable_model_parameters(model: Union[PreTrainedModel, TFPreTrainedModel]):
        trainable_model_params = 0
        all_model_params = 0
        for _, param in model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        return print(f"trainable model parameters: {trainable_model_params}\n"\
                    f"all model parameters: {all_model_params}\n"\
                    f"percentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%")

    @staticmethod
    def _get_bits_and_bytes_config(bnb_config_params):
        bnb_config = BitsAndBytesConfig(
            **bnb_config_params
        )
        return bnb_config