import sys
import torch
from transformers import (
  AutoConfig,
  AutoTokenizer, 
  AutoModelForCausalLM, 
  BitsAndBytesConfig,
  pipeline
)

from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncChromiumLoader

from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore

import nest_asyncio
#################################################################
# Tokenizer
#################################################################

model_name='mistralai/Mistral-7B-Instruct-v0.1'
sentence_transformer = 'BAAI/bge-large-en-v1.5'

model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

model_config = AutoConfig.from_pretrained(
    model_name,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#################################################################
# bitsandbytes parameters
#################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

#################################################################
# Set up quantization config
#################################################################
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

#################################################################
# Load pre-trained config
#################################################################
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(model))

# pipeline params in below file
# /home/atul/.local/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:40
text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=10000,
    do_sample=True
)

mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

nest_asyncio.apply()

# Articles to index
articles = ["https://www.fantasypros.com/2023/11/rival-fantasy-nfl-week-10/",
            "https://www.fantasypros.com/2023/11/5-stats-to-know-before-setting-your-fantasy-lineup-week-10/",
            "https://www.fantasypros.com/2023/11/nfl-week-10-sleeper-picks-player-predictions-2023/",
            "https://www.fantasypros.com/2023/11/nfl-dfs-week-10-stacking-advice-picks-2023-fantasy-football/",
            "https://www.fantasypros.com/2023/11/players-to-buy-low-sell-high-trade-advice-2023-fantasy-football/"]

# Scrapes the blogs above
# loader = AsyncChromiumLoader(articles)
# docs = loader.load()

# # Converts HTML to plain text 
# html2text = Html2TextTransformer()
# docs_transformed = html2text.transform_documents(docs)

# Chunk text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=10, separators=['\n\n', '\n', '.']
)
# text_splitter = CharacterTextSplitter(chunk_size=100, 
#                                       chunk_overlap=0)
# chunked_documents = text_splitter.split_documents(docs_transformed)

# # embedding_model = SentenceTransformerEmbeddings(model_name=sentence_transformer)
embedding_model = HuggingFaceEmbeddings(model_name=sentence_transformer,
                                        model_kwargs=model_kwargs,
                                        encode_kwargs=encode_kwargs)

# import pdb; pdb.set_trace()

print("Connecting DB...")
vector_db = Milvus(embedding_function=embedding_model, collection_name="e4r_doc",
                   connection_args={"host": "127.0.0.1", "port": "19530"})

# Load chunked documents into the Milvus
# vector_db = Milvus.from_documents(
#     chunked_documents,
#     embedding_model,
#     connection_args={"host": "127.0.0.1", "port": "19530"},
# )
# # for docs in chunked_documents:
# #     _ = vector_db.add_documents([docs])
# retriever = vector_db.as_retriever()

fs = LocalFileStore("/home/atul/Documents/langchain/db/docStore")
store = create_kv_docstore(fs)

retriever = ParentDocumentRetriever(
            vectorstore=vector_db,
            docstore=store,
            child_splitter=text_splitter,
            parent_splitter=text_splitter,
        )

# retriever.add_documents(chunked_documents)

# create our Q&A chain
pdf_qa = ConversationalRetrievalChain.from_llm(
    mistral_llm,
    retriever=retriever,
    return_source_documents=True,
    verbose=True
)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

chat_history = []
print(f"{yellow}---------------------------------------------------------------------------------")
print('Welcome to the DocBot. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')
while True:
    query = input(f"{green}Prompt: ")
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = pdf_qa.invoke(
        {"question": query, "chat_history": chat_history})
    print(f"{white}Answer: " + result["answer"])
    chat_history.append((query, result["answer"]))