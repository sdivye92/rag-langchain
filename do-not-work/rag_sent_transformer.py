import torch
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS, LanceDB
from langchain.chains import RetrievalQA
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from transformers import (
  AutoTokenizer, 
  AutoModelForCausalLM, 
  BitsAndBytesConfig
)
import transformers

from dummy.prompt import *

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


chroma_db = Chroma.from_documents(document_chunks, embedding_model, persist_directory="./db/")
retriever = chroma_db.as_retriever()
import pdb; pdb.set_trace()

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

# # Prompt template 
# qa_template = """<s>[INST] You are a helpful assistant.
# Use the following context to Answer the question below briefly:

# {context}

# {question} [/INST] </s>
# """

# # Create a prompt instance 
# QA_PROMPT = PromptTemplate(input_variables=['context', 'question'],
#                output_parser=None,
#                partial_variables={},
#                template=qa_template,
#                template_format='f-string',
#                validate_template=True
#                )

# # Custom QA Chain 
# qa_chain = RetrievalQA.from_chain_type(
#     model,
#     retriever=retriever,
#     chain_type_kwargs={"prompt": QA_PROMPT}
# )

# # Your Question 
# question = query #"YOUR QUESTION HERE"

# # Query Mistral 7B Instruct model
# response = qa_chain({"query": question})

# # Print your result 
# print(response['result'])