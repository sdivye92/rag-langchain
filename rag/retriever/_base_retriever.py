from typing import List
from abc import ABC, abstractmethod
from langchain.docstore.document import Document

class Retriever(ABC):
    
    @abstractmethod
    def add_documents(self, documents: List[Document],
                      ids: List[str] | None = None,
                      add_to_docstore: bool = True):
        pass