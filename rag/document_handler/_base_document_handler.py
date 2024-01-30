from glob import glob
import imp
import logging
import os
from typing import List
from abc import ABC, abstractmethod

import pandas as pd
from rag.config import Config
from rag.retriever import Retriever
from pymilvus import connections, Collection
from langchain.docstore.document import Document

class DocumentHandler(ABC):
    def __init__(self, dir_path, retrierver: Retriever, config: Config):
        self.retriever=retrierver
        self.config=config
        self.dir_path=dir_path
        self.file_type=None
        self.files_set=set([])
    
    @abstractmethod
    def _get_doc_from_file(self, file: str) -> List[Document]:
       pass
    
    def load_documents(self, files: List[str]):
        for file in files:
            try:
                logging.info(f"Trying to add file to vectorstore: {file}")
                docs = self._get_doc_from_file(file)
                self.add_documents_to_database(docs)
                self.files_set.add(file)
                logging.info(f"File added successfully to vectorstore: {file}")
            except Exception as e:
                logging.exception(e)
    
    def remove_documents(self, files: List[str]):
        for file in files:
            try:
                logging.info(f"Trying to remove file from vectorstore: {file}")
                self.delete_document_from_database(file)
                self.files_set.remove(file)
                logging.info(f"File removed successfully from vectorstore: {file}")
            except Exception as e:
                logging.exception(e)
    
    def add_documents_to_database(self, doc_list: List[Document]):
        self.retriever.add_documents(doc_list)
    
    def delete_document_from_database(self, file: str):
        _ = connections.connect(**self.config.milvus.connection_args)
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
        connections.disconnect("default")
        
    
    def update_database(self):
        import pdb; pdb.set_trace()
        files_in_dir = set(glob(os.path.join(self.dir_path,f"**/*.{self.file_type}"), recursive=True))
        if files_to_add:=(files_in_dir - self.files_set):
            self.load_documents(files_to_add)
        if files_to_remove:=(self.files_set - files_in_dir):
            self.remove_documents(files_to_remove)