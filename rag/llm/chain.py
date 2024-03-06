from typing import Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import MultiVectorRetriever
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from rag.config import Config
from rag.retriever import Retriever
from langchain.memory import ConversationBufferMemory
class Chain:
    def __init__(self, pipeline: HuggingFacePipeline,
                 prompt: PromptTemplate, retriever: MultiVectorRetriever,
                 memory: Optional[ConversationBufferMemory]=None,
                 verbose: bool=False):
        # self.config=config
        self.pipeline=pipeline
        self.prompt=prompt
        self.retriever=retriever
        self.memory = memory
        self.verbose=verbose
        self.llm_chain = self._get_llm_chain(self.prompt)
        self.handle_empty_context_lambda=RunnableLambda(self._handle_empty_context)
    
    def _get_llm_chain(self, prompt: PromptTemplate):
        if self.memory is not None:
            return LLMChain(llm=self.pipeline, prompt=prompt, memory=self.memory, verbose=self.verbose)
        return LLMChain(llm=self.pipeline, prompt=prompt, verbose=self.verbose)
    
    def _handle_empty_context(self, data):
        # import pdb; pdb.set_trace()
        context, question = data['context'], data['question']
        if not context:
            return {"context": f"you don't know the answer to the question: {question}. So just say that you do not know the answer given the context.", "question": question}
        else:
            return data
    
    def invoke(self, question: str):
        rag_chain = ( 
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.handle_empty_context_lambda
            | self.llm_chain
        )
        # import pdb; pdb.set_trace()
        result = rag_chain.invoke(question)
        return result