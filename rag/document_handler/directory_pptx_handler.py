import os
import logging
import pandas as pd
from glob import glob
from rag.config import Config
from ._base_document_handler import DocumentHandler
from rag.retriever import Retriever
from langchain_community.document_loaders import UnstructuredPowerPointLoader

class DirectoryPptxHandler(DocumentHandler):
    FILE_TYPE = "pptx"
    def __init__(self, dir_path, retriever: Retriever, config: Config):
        super().__init__(dir_path, retriever, config)
        self._metadata_tags = ['source', 'file_directory', 'filename', 'category']
    
    def _get_doc_from_file(self, file):
        pptx_loader = UnstructuredPowerPointLoader(file)
        docs = pptx_loader.load()
        return docs