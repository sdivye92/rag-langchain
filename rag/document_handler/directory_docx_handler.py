from rag.config import Config
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from ._base_document_handler import DocumentHandler
from rag.retriever import Retriever

class DirectoryDocxHandler(DocumentHandler):
    def __init__(self, dir_path, retriever: Retriever, config: Config):
        super().__init__(dir_path, retriever, config)
        self.file_type = "docx"
        self._metadata_tags = ['source', 'file_directory', 'filename', 'category']
    
    def _get_doc_from_file(self, file):
        docx_loader = UnstructuredWordDocumentLoader(file, mode="elements")
        docs = docx_loader.load()
        for doc in docs:
            metadata = doc.metadata
            metadata_filtered = {tag: metadata[tag] for tag in self._metadata_tags}
                    # import pdb; pdb.set_trace()
            if links:=metadata.get('links', []):
                metadata_filtered['link'] = {link['text']: link for link in links}
            else:
                metadata_filtered['link'] = {}
            doc.metadata = metadata_filtered
        return docs