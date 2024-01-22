from typing import List
from abc import ABC, abstractmethod
from langchain.docstore.document import Document

class DocumentHandler(ABC):
    
    def __init__(self, retrierver):
        self.retriever=retrierver
    
    @abstractmethod
    def load_documents(self, files: List[str]):
        pass
    
    @abstractmethod
    def remove_documents(self, files: List[str]):
        pass
    
    @abstractmethod
    def update_database(self):
        pass
    
    @abstractmethod
    def add_documents_to_database(self, doc_list: List[Document]):
        pass
    
    @abstractmethod
    def delete_document_from_database(self, file: str):
        pass