import json
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
    error_msg = "Invalid collection name. Collection name can only have letters, digits and underscore."
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
        if set(self.chunk_collection_name + self.doc_collection_name)-self.allowed:
            raise ValueError(self.error_msg)



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
        llm_model="mistralai/Mistral-7B-Instruct-v0.1"

        self.doc_store_path = "../docStore"


        config = {}
        with open(config_path, 'r') as config_f:
            config = json.load(config_f)
        
        embedding = config.get('embedding', {})
        milvus = config.get('milvus', {})

        self.embedding = EmbeddingConfig(embedding)
        self.milvus = MilvusConfig(milvus)


    @classmethod
    def init(cls, config_path):
        cls.__instance = Config(config_path)
    
    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            raise Exception('Instance not initialized, use Config.init to initialize the config')

        return cls.__instance