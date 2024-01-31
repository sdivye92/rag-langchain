import os
import json
import torch
import string

class EmbeddingConfig(dict):
    def __init__(self, embedding_config):
        self.model_name="BAAI/bge-large-en-v1.5"
        self.model_kwargs={'device': 'cpu'}
        self.encode_kwargs={'normalize_embeddings': True}

        self.__dict__.update(embedding_config)
        super().__init__(self.__dict__)

class MilvusConfig(dict):
    allowed = set(string.ascii_letters + string.digits + "_")
    error_msg = "Invalid collection name '{}'. Collection name can only have letters, digits and underscore."
    def __init__(self, milvus_config):
        self.chunk_collection_name="ChunkStore"
        self.connection_args={
            "host": "127.0.0.1",
            "port": "19530"
        }
        self.__dict__.update(milvus_config)
        self.__validate_collection_names()
        super().__init__(self.__dict__)
    
    def __validate_collection_names(self):
        if set(self.chunk_collection_name)-self.allowed:
            raise ValueError(self.error_msg.format(self.chunk_collection_name))

class LLMConfig:
    def __init__(self, llm_config):
        self.model_name='mistralai/Mistral-7B-Instruct-v0.1'
        self.params = {
            "temperature":0.2,
            "repetition_penalty":1.1,
            "return_full_text":True,
            "max_new_tokens":10000,
            "do_sample":True
        }
        self.__dict__.update(llm_config)
        
class BitsAndBytesConfig(dict):
    def __init__(self, bnb_config):
        # Activate 4-bit precision base model loading
        use_4bit = True
        # Compute dtype for 4-bit base models
        bnb_4bit_compute_dtype = "float16"
        # Quantization type (fp4 or nf4)
        bnb_4bit_quant_type = "nf4"
        # Activate nested quantization for 4-bit base models (double quantization)
        use_nested_quant = False
        # Set up quantization config
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        
        self.load_in_4bit=use_4bit
        self.bnb_4bit_quant_type=bnb_4bit_quant_type
        self.bnb_4bit_compute_dtype=compute_dtype
        self.bnb_4bit_use_double_quant=use_nested_quant
        
        self.__dict__.update(bnb_config)
        super().__init__(self.__dict__)

class Config:
    __instance = None

    def __init__(self, config_path):
        self.parent_splitter={
            "chunk_size" : 2000,
            "chunk_overlap" : 200,
            "separators" : ['\n\n', '\n', '.', ' ', '']
        }
        self.child_splitter={
            "chunk_size" : 500,
            "chunk_overlap" : 10,
            "separators" : ['\n\n', '\n', '.', ' ', '']
        }

        self.doc_store_path = "../docStore"
        
        self.supported_file_types = [
            "pdf", "docx"
        ]


        config = {}
        config_path = os.path.abspath(os.path.expanduser(config_path))
        with open(config_path, 'r') as config_f:
            config = json.load(config_f)
            
        base_config = {k:v for k, v in config.items() 
                       if k not in ['milvus', 'embedding']}
        
        self.__dict__.update(base_config)
        
        embedding = config.get('embedding', {})
        milvus = config.get('milvus', {})
        bnb = config.get('bits_and_bytes', {})
        llm = config.get('llm', {})

        self.embedding = EmbeddingConfig(embedding)
        self.milvus = MilvusConfig(milvus)
        self.llm = LLMConfig(llm)
        self.bits_and_bytes = BitsAndBytesConfig(bnb)


    @classmethod
    def init(cls, config_path):
        cls.__instance = Config(config_path)
    
    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            raise Exception('Instance not initialized, use Config.init to initialize the config')

        return cls.__instance