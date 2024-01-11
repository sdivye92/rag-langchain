import torch
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain.chains import RetrievalQA
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from transformers import (
  AutoTokenizer, 
  AutoModelForCausalLM, 
  BitsAndBytesConfig
)
import transformers

from prompt import *

sentence_transformer = 'BAAI/bge-large-en-v1.5'
llm_model_name='mistralai/Mistral-7B-Instruct-v0.1'

model_config = transformers.AutoConfig.from_pretrained(
    llm_model_name
)

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    quantization_config=bnb_config,
)


docs = [Document(page_content=post) for post in [persona_prompt, audience_persona]]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=10, separators=['\n\n', '\n', '.']
)

document_chunks = text_splitter.split_documents(docs)

embedding_model = SentenceTransformerEmbeddings(model_name=sentence_transformer)
import pdb; pdb.set_trace()

vector_db = Milvus.from_documents(
    docs,
    embedding_model,
    connection_args={"host": "127.0.0.1", "port": "19530"},
)
retriever = vector_db.as_retriever()


query = "When was the last time you thought a mechanic would help you with your car issue?"
# docs = chroma_db.similarity_search(query, k=1)
# print(docs)



tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

inputs_not_chat = tokenizer.encode_plus("[INST] Explain in detail about reinforcement learning? [/INST]", return_tensors="pt")['input_ids'].to('cuda')

generated_ids = model.generate(inputs_not_chat, 
                               max_new_tokens=10000, 
                               do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)

print(decoded)