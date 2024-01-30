from rag.config import Config
from rag.retriever import ParentChildRetriever
from rag.db_connection_handler import MilvusConnectorBuilder
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

sentence_transformer = 'BAAI/bge-large-en-v1.5'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

config = Config("./config.json")
config.milvus.chunk_collection_name = "e4r_doc"

embedding_model = HuggingFaceEmbeddings(model_name=sentence_transformer,
                                        model_kwargs=model_kwargs,
                                        encode_kwargs=encode_kwargs)

mcb = MilvusConnectorBuilder(config.milvus.connection_args, embedding_model)
pcr = ParentChildRetriever(config, mcb)

loader = UnstructuredWordDocumentLoader("../e4r_docs/E4R SOQ/2022/SOQ -Q1-2022 Accelerated computing/E4R SOQ- DIC.docx", mode="elements")
data2 = loader.load()

metadata_tags = ['source', 'file_directory', 'filename', 'category']

for dd in data2:
    # dd = data2[2]
    md = dd.metadata
    md2 = {tag: md[tag] for tag in metadata_tags}
    # import pdb; pdb.set_trace()
    if links:=md.get('links', []):
        md2['link'] = {link['text']: link for link in links}
    else:
        md2['link'] = {}
    dd.metadata = md2

pcr.add_documents(data2)