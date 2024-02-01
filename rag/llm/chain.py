from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import MultiVectorRetriever
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from rag.config import Config
from rag.retriever import Retriever

class Chain:
    def __init__(self, pipeline: HuggingFacePipeline,
                 prompt: PromptTemplate, retriever: MultiVectorRetriever):
        # self.config=config
        self.pipeline=pipeline
        self.prompt=prompt
        self.retriever=retriever
        self.llm_chain = self._get_llm_chain(self.prompt)
        self.handle_empty_context_lambda=RunnableLambda(self._handle_empty_context)
    
    def _get_llm_chain(self, prompt: PromptTemplate):
        return LLMChain(llm=self.pipeline, prompt=prompt)
    
    def _handle_empty_context(self, data):
        context, question = data['context'], data['question']
        print(data)
        if not context:
            print("i'm here")
            return {"context": f"you don't know the answer to the question: {question}.\
                So just say that you do not know the answer given the context", "question": question}
        else:
            print("no, i'm here")
            return data
    
    def invoke(self, question: str):
        rag_chain = ( 
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.handle_empty_context_lambda
            | self.llm_chain
        )
        import pdb; pdb.set_trace()
        result = rag_chain.invoke(question)
        return result