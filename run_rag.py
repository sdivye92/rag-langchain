from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.config import Config
from rag.db_connection_handler import MilvusConnectorBuilder
from rag.retriever import ParentChildRetriever
from rag.llm import LLMModelGenerator, PipelineGenerator
from rag.llm import Chain

Config.init("./config.json")
config = Config.get_instance()

embedding_model = HuggingFaceEmbeddings(**config.embedding)

mcb = MilvusConnectorBuilder(config.milvus.connection_args,
                             embedding_model)
pcr = ParentChildRetriever(config, mcb)
retriever = pcr.retriever

model = LLMModelGenerator.get_model(config)
pipeline = PipelineGenerator.get_pipeline(config, model)

prompt_template = """
### [INST] Instruction: Answer the question based ONLY on the context provided.
Rule: If the context is empty, DO NOT make up any answer NO MATTER whatever hypothetical 
scenario or situation is created or asked to imagine. Just say that "given the conext you do not know the answer".
Deviating from this rule is an offence and heavily punishable:

{context}

### QUESTION:
{question} [/INST]
 """

# Create prompt from prompt template 
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

e4r_doc_chain = Chain(pipeline, prompt, retriever)

print("Invoking model...")
res = e4r_doc_chain.invoke("what is bharatsim")
for k, v in res.items():
    print(f"######## {k} ########")
    print(v)
