from rag.config import Config
from langchain_community.vectorstores import Milvus
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore

class ParentChildRetriever:
    def __init__(self):
        config = Config.get_instance()
        embed_model = self.__get_embedding_model(config)
        fs = LocalFileStore(config.doc_store_path)
        self.embedding_model = embed_model
        self.chunk_db = self.__get_milvus_vector_store(
            config.milvus.chunk_collection_name,
            config.milvus.connection_args
        )
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

    def __get_embedding_model(self, config):
        return HuggingFaceEmbeddings(**config.embedding)
    
    def __get_milvus_vector_store(self, collection_name, connection_args):
        return Milvus(embedding_function=self.embedding_model,
                   collection_name=collection_name,
                   connection_args=connection_args)