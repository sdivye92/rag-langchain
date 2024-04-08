import os
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from rag.config import Config
from rag.db_connection_handler import MilvusConnectorBuilder
from rag.retriever import ParentChildRetriever
from rag.llm import LLMModelGenerator, PipelineGenerator
from rag.llm import Chain, Gemini, OllamaMistral

Config.init("./config.json")
config = Config.get_instance()

embedding_model = HuggingFaceEmbeddings(**config.embedding)

mcb = MilvusConnectorBuilder(config.milvus.connection_args,
                             embedding_model)
pcr = ParentChildRetriever(config, mcb)
retriever = pcr.retriever

# model = LLMModelGenerator.get_model(config)
# pipeline = PipelineGenerator.get_pipeline(config, model)

# pipeline = Gemini.get_llm(google_api_key=os.getenv("GOOGLE_API_KEY"), 
#                     verbose=True, temperature=0.5)

pipeline = OllamaMistral.get_llm('http://localhost:11434')

prompt_template = """
### [INST] Instruction: Answer the question based ONLY on the context provided.
Rule: If the context is empty, DO NOT make up any answer NO MATTER whatever hypothetical 
scenario or situation is created or asked to imagine. Just say that "given the conext you do not know the answer".
Deviating from this rule is an offence and heavily punishable:

{context}

{chat_history}

### QUESTION:
{question} [/INST]
 """
memory_key = "chat_history"
input_key = "question"
# Create prompt from prompt template 
prompt = PromptTemplate(
    input_variables=["context", input_key, memory_key],
    template=prompt_template,
)
memory = ConversationBufferMemory(memory_key=memory_key,
                                  input_key=input_key)
e4r_doc_chain = Chain(pipeline, prompt, retriever, memory)

print("Invoking model...")
while True:
    question = input("Human:\n")
    if question.strip().lower() in ["exit", "quit"]:
        break
    else:
        res = e4r_doc_chain.invoke(question)
        print(f"######## history ########")
        print(res['context'])
        print(f"######## history ########")
        print(f"AI:\n{res['text'].strip()}\n")
        # for k, v in res.items():
        #     print(f"######## {k} ########")
        #     print(v)
