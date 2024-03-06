from typing import List, Union
from abc import ABC, abstractmethod
from langchain_core.stores import BaseStore
from langchain_community.vectorstores import VectorStore
from langchain.docstore.document import Document

class Retriever(ABC):
    def __init__(self):
        self.retriever=None
        self.doc_db: Union[BaseStore, None]=None
        self.chunk_db: Union[VectorStore, None]=None
    
    @abstractmethod
    def add_documents(self, documents: List[Document],
                      ids: List[str] | None = None,
                      add_to_docstore: bool = True):
        pass