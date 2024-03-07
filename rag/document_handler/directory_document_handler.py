import os
import logging
from enum import Enum
from typing import List, Literal
from rag.document_handler import (
    DocumentHandler,
    DirectoryDocxHandler,
    DirectoryPDFHandler,
    DirectoryPptxHandler
)
from rag.config import Config
from rag.retriever import Retriever

FileTypes = {
    "pdf": DirectoryPDFHandler,
    "docx": DirectoryDocxHandler,
    "pptx": DirectoryPptxHandler
}

class DirectoryDocumentHandler:
    def __init__(self, dir_path, retriever: Retriever, config: Config):
        self.dir_path = os.path.abspath(os.path.expanduser(dir_path))
        self.retriever = retriever
        self.config = config
        try:
            self.file_type_handlers: List[DocumentHandler] = [
                FileTypes[file_type](self.dir_path, self.retriever, self.config)
                    for file_type in config.supported_file_types
            ]
        except Exception as e:
            logging.exception(e)
        
    def update_database(self):
        for file_type_handler in self.file_type_handlers:
            file_type_handler.update_database()