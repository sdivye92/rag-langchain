from rag.config import Config
from langchain_community.vectorstores import Milvus
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore

config = Config.get_instance()

class ParentChildRetiever:
    def __init__(self):
        embed_model = self.__get_embedding_model()
        self.embedding_model = embed_model
        self.chunk_db = self.__get_milvus_vector_store(
            config.milvus.doc_collection_name,
            config.milvus.connection_args
        )
        self.doc_db = self.__get_milvus_vector_store(
            config.milvus.doc_collection_name,
            config.milvus.connection_args
        )
        self.parent_splitter = RecursiveCharacterTextSplitter(**config.parent_splitter)
        self.child_splitter = RecursiveCharacterTextSplitter(**config.child_splitter)
        fs = LocalFileStore(config.doc_store_path)
        store = create_kv_docstore(fs)

        self.retriever = ParentDocumentRetriever(
            vectorstore=self.chunk_db,
            docstore=store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )

    def __get_embedding_model(self):
        return HuggingFaceEmbeddings(**config.embedding)
    
    def __get_milvus_vector_store(self, collection_name, connection_args):
        return Milvus(embedding_function=self.embedding_model,
                   collection_name=collection_name,
                   connection_args=connection_args)