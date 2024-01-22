from rag.config import Config
from langchain_community.vectorstores import Milvus
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore
from rag.db_connection_handler import OffMemoryDatabaseConnectorBuilder

class ParentChildRetriever:
    def __init__(self, config: Config, db_connector_builder: OffMemoryDatabaseConnectorBuilder):
        embed_model = self.__get_embedding_model(config)
        fs = LocalFileStore(config.doc_store_path)
        self.embedding_model = embed_model
        self.chunk_db = db_connector_builder.get_connector(
            config.milvus.chunk_collection_name)
        # based on https://stackoverflow.com/a/77397998
        self.doc_db = create_kv_docstore(fs)
        self.parent_splitter = RecursiveCharacterTextSplitter(**config.parent_splitter)
        self.child_splitter = RecursiveCharacterTextSplitter(**config.child_splitter)

        self.retriever = ParentDocumentRetriever(
            vectorstore=self.chunk_db,
            docstore=self.doc_db,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )

    def __get_embedding_model(self, config: Config):
        return HuggingFaceEmbeddings(**config.embedding)