import logging
from rag.config import Config
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.db_connection_handler import MilvusConnectorBuilder
from rag.document_handler.webpage_document_handler import WebpageDocumentHandler
from rag.retriever import ParentChildRetriever
from rag.document_handler import DirectoryDocumentHandler

logging_format = "[%(filename)s:%(lineno)s - %(funcName)s()] [%(levelname)s] :: %(message)s"
logging.basicConfig(format=logging_format, datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

Config.init("./config.json")
config = Config.get_instance()

webpages = ["https://www.thoughtworks.com/en-in/clients/engineering-research",
            "https://www.thoughtworks.com/en-in/clients/engineering-research/bharatsim",
            "https://www.thoughtworks.com/en-in/clients/engineering-research/artip",
            "https://www.thoughtworks.com/en-in/clients/engineering-research/tmt",
            "https://www.thoughtworks.com/en-in/clients/engineering-research/star-formation-histories",
            "https://www.thoughtworks.com/en-in/clients/engineering-research/solar-radio-imaging",
            "https://www.thoughtworks.com/en-in/clients/engineering-research/perc"]

embedding_model = HuggingFaceEmbeddings(model_name=config.embedding.model_name,
                                        model_kwargs=config.embedding.model_kwargs,
                                        encode_kwargs=config.embedding.encode_kwargs)
mcb = MilvusConnectorBuilder(config.milvus.connection_args, embedding_model)
pcr = ParentChildRetriever(config, mcb)
directory_doc_handler = DirectoryDocumentHandler("~/Documents/e4r_docs", pcr, config)
webpage_doc_handler = WebpageDocumentHandler("~/Documents/e4r_docs", pcr, config)

webpage_doc_handler.load_documents(webpages)
directory_doc_handler.update_database()