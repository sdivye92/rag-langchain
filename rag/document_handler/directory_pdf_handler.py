import re
from rag.config import Config
from rag.retriever import Retriever
from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader
from langchain_community.document_transformers import Html2TextTransformer
from ._base_document_handler import DocumentHandler

class DirectoryPDFHandler(DocumentHandler):
    FILE_TYPE = "pdf"
    def __init__(self, dir_path, retriever: Retriever, config: Config):
        super().__init__(dir_path, retriever, config)
    
    def _get_doc_from_source(self, source):
        pdf_html_loader = PDFMinerPDFasHTMLLoader(source)
        html = pdf_html_loader.load()[0]
        html2text = Html2TextTransformer()
        docs = html2text.transform_documents([html])
        for doc in docs:
            doc.page_content = re.sub(r"(\n\n)", "\n",
                                              re.sub(r"(?<!\n)\n(?!\n)", "",
                                                     doc.page_content))
                                             
        return docs