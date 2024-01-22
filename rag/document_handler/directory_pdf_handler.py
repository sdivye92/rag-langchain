import pandas as pd
from pymilvus import connections, Collection
from glob import glob
import os
from langchain_community.document_loaders import PyMuPDFLoader
from ._base_document_handler import DocumentHandler
import logging
from rag.config import Config

class DirectoryPDFHandler(DocumentHandler):
    def __init__(self, dir_path, retriever, config: Config):
        super().__init__(retriever)
        self.config=config
        self.dir_path=dir_path
        self.pdf_set=set([])
    
    def load_documents(self, files):
        for file in files:
            try:
                docs = PyMuPDFLoader(file)
                self.add_documents_to_database(docs)
            except Exception as e:
                logging.error(e)
    
    def remove_documents(self, files):
        for file in files:
            try:
                self.delete_document_from_database(file)
            except Exception as e:
                logging.error(e)
    
    def add_documents_to_database(self, doc_list):
        self.retriever.retriever.add_document(doc_list)
    
    def delete_document_from_database(self, file):
        conn = connections.connect(**self.config.milvus.connection_args)
        collection = Collection(self.config.milvus.chunk_collection_name)
        expr = f"file_path == '{file}'"
        output_fields=["doc_id"]
        batch_size = 10
        limit = -1
        query_iterator = collection.query_iterator(batch_size, limit, expr, output_fields)
        result = []
        while True:
            res = query_iterator.next()
            if len(res) == 0:
                print("query iteration finished, close")
                # close the iterator
                query_iterator.close()
                break
            result.extend(res)
        doc_ids = pd.DataFrame(result)\
                    .drop_duplicates('doc_id', ignore_index=True)['doc_id'].to_list()
        self.retriever.doc_db.mdelete(doc_ids)
        collection.delete(expr=expr)
        
    
    def update_database(self):
        files_in_dir = set(glob(os.path.join(self.dir_path,"*.pdf")))
        if files_to_add:=(files_in_dir - self.pdf_set):
            self.load_documents(files_to_add)
        if files_to_remove:=(self.pdf_set - files_in_dir):
            self.remove_documents(files_to_remove)