from rag.config import Config
from langchain.storage import LocalFileStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage._lc_store import create_kv_docstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag.db_connection_handler import OffMemoryDatabaseConnectorBuilder
from ._base_retriever import Retriever

class ParentChildRetriever(Retriever):
    def __init__(self, config: Config, db_connector_builder: OffMemoryDatabaseConnectorBuilder):
        fs = LocalFileStore(config.doc_store_path)
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
    
    def add_documents(self, documents, ids=None, add_to_docstore=True):
        self.retriever.add_documents(documents, ids, add_to_docstore)