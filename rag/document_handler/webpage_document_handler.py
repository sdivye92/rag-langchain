from typing import Iterable, List
from langchain.docstore.document import Document
import nest_asyncio
from rag.config import Config
from rag.retriever import Retriever
from ._base_document_handler import DocumentHandler
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncChromiumLoader

class WebpageDocumentHandler(DocumentHandler):
    def __init__(self, dir_path, retriever: Retriever, config: Config):
        super().__init__(dir_path, retriever, config)
    
    
    def _get_doc_from_source(self, source: str):
        loader = AsyncChromiumLoader([source])
        docs = loader.load()

        # Converts HTML to plain text 
        html2text = Html2TextTransformer()
        docs = html2text.transform_documents(docs)
        return docs
    
    def update_database(self):
        pass