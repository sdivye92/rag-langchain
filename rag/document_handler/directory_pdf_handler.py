import re
import os
import logging
import pandas as pd
from glob import glob
from rag.config import Config
from rag.retriever import Retriever
from pymilvus import connections, Collection
from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader
from langchain_community.document_transformers import Html2TextTransformer
from ._base_document_handler import DocumentHandler

class DirectoryPDFHandler(DocumentHandler):
    def __init__(self, dir_path, retriever: Retriever, config: Config):
        super().__init__(dir_path, retriever, config)
        self.file_type = "pdf"
    
    def _get_doc_from_file(self, file):
        pdf_html_loader = PDFMinerPDFasHTMLLoader(file)
        html = pdf_html_loader.load()[0]
        html2text = Html2TextTransformer()
        docs = html2text.transform_documents([html])
        for doc in docs:
            doc.page_content = re.sub(r"(\n\n)", "\n",
                                              re.sub(r"(?<!\n)\n(?!\n)", "",
                                                     doc.page_content))
                                             
        return docs