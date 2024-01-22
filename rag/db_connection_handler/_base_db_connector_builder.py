from abc import ABC, abstractmethod

class OffMemoryDatabaseConnectorBuilder(ABC):

    @abstractmethod
    def get_connector(self, collection_name):
        pass
    
    @abstractmethod
    def set_embedding_model(self, embedding_model):
        pass