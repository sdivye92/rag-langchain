import logging
from rag.config import Config
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.db_connection_handler import MilvusConnectorBuilder
from rag.retriever import ParentChildRetriever
from rag.document_handler import DirectoryDocumentHandler

logging_format = "[%(filename)s:%(lineno)s - %(funcName)s()] [%(levelname)s] :: %(message)s"
logging.basicConfig(format=logging_format, datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

Config.init("./config.json")
config = Config.get_instance()

embedding_model = HuggingFaceEmbeddings(model_name=config.embedding.model_name,
                                        model_kwargs=config.embedding.model_kwargs,
                                        encode_kwargs=config.embedding.encode_kwargs)
mcb = MilvusConnectorBuilder(config.milvus.connection_args, embedding_model)
pcr = ParentChildRetriever(config, mcb)
directory_doc_handler = DirectoryDocumentHandler("~/Documents/e4r_docs/E4R SOQ", pcr, config)

directory_doc_handler.update_database()