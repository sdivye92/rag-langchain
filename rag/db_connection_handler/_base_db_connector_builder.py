from langchain_community.vectorstores import VectorStore
from abc import ABC, abstractmethod

class OffMemoryDatabaseConnectorBuilder(ABC):

    @abstractmethod
    def get_connector(self, collection_name) -> VectorStore:
        pass
    
    @abstractmethod
    def set_embedding_model(self, embedding_model):
        pass