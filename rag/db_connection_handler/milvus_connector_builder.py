from langchain_community.vectorstores.milvus import Milvus
from ._base_db_connector_builder import OffMemoryDatabaseConnectorBuilder

class MilvusConnectorBuilder(OffMemoryDatabaseConnectorBuilder):
    def __init__(self, connection_args, embedding_model):
        self.connection_args = connection_args
        self.embedding_model=embedding_model
        self.__connector_list = {}

    def get_connector(self, collection_name) -> Milvus:
        if collection_name in self.__connector_list:
            return self.__connector_list[collection_name]
        else:
            connector = Milvus(embedding_function=self.embedding_model,
                    collection_name=collection_name,
                    connection_args=self.connection_args)
            self.__connector_list[collection_name] = connector
        return connector

    def set_embedding_model(self, embedding_model):
        self.embedding_model=embedding_model