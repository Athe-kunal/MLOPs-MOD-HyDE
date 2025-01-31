import openai
from dotenv import load_dotenv,find_dotenv
import os
import dspy
from mod_hyde.mod_hyde.config import *
from transformers import pipeline, AutoTokenizer
from rerankers import Reranker
import torch
import wandb
import shutil

load_dotenv(find_dotenv(),override=True)
openai.api_key = os.environ["OPENAI_API_KEY"]
llm = dspy.OpenAI(model="gpt-3.5-turbo-0125",max_tokens = 4096)
dspy.settings.configure(lm=llm)
class GenerateAnswer(dspy.Signature):
    """Answer questions in detail based on the context."""
    
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    reasoning = dspy.OutputField(desc='Reasoning behind the answer')
    answer = dspy.OutputField(desc="answer in detail. 'Answer: ' should strictly come after 'Reasoning: '")

# gpt3_hyde = dspy.OpenAI(model=HYDE_MODEL, max_tokens=300)
gpt3_hyde = dspy.Google(api_key=os.environ['GOOGLE_API_KEY'],max_output_tokens=300)

class HyDEGenerateAnswer(dspy.Signature):
    """Answer the question concisely and include as many details as possible"""
    question = dspy.InputField()
    reasoning = dspy.OutputField(desc='Reasoning behind the answer')
    answer = dspy.OutputField(desc="answer concisely with as many details as possible")

if os.path.exists("artifacts"):
    shutil.rmtree("artifacts")

run = wandb.init()
artifact = run.use_artifact('WGQA_T5/model-registry/Modified HyDE model collection:v0', type='model')
artifact_dir = artifact.download()

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
model_name = os.listdir("artifacts")[0]
pipe = pipeline("text-generation", model=model_name,tokenizer=tokenizer)
def get_mod_HyDE_answer(question):
    out = pipe(question,max_new_tokens=300,do_sample=True,min_new_tokens=10)
    return out[0]['generated_text']


class RAG(dspy.Module):
    def __init__(self,retriever,use_reranker:bool=False,use_cot:bool=True,rerank_docs:int=5):
        super().__init__()
        ret = retriever.retrieve("Explore the significance of valuation")

        assert ret[0].text != "", "The retriever is not working properly"
        self.use_reranker = use_reranker
        if self.use_reranker:
            assert rerank_docs>0, "If you are using re-ranker, then please provide more than 0 rerank_docs"
            if torch.cuda.is_available():
                self.ranker = Reranker("colbert",device='cuda')
            else:
                self.ranker = Reranker("colbert",device='cpu')

        self.rerank_docs = rerank_docs
        self.retrieve_model = retriever
        self.hyde_answer = dspy.Predict(HyDEGenerateAnswer)
        if use_cot:
            self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        else:
            self.generate_answer = dspy.Predict(GenerateAnswer)
    
    def retrieve(self,query:str):
        llama_index_docs = self.retrieve_model.retrieve(query)
        metadata_list = []
        context_list = []

        for ld in llama_index_docs:
            metadata_list.append(ld.metadata)
            context_list.append(ld.text)
        return context_list,metadata_list
    def forward(self, question,algo_type:str="frozen"):
        assert algo_type in ["frozen","hyde","mod_hyde"], 'The algo type should be from ["frozen","hyde","mod-hyde"]'
        if algo_type == 'hyde':
            with dspy.context(lm=gpt3_hyde):
                hyde_answer = self.hyde_answer(question=question).answer
            context,metadata = self.retrieve(hyde_answer)
            if self.use_reranker: results = self.ranker.rank(query=hyde_answer, docs=context, doc_ids=[i for i in range(len(context))])
        elif algo_type == 'frozen':
            context,metadata = self.retrieve(question)
            if self.use_reranker: results = self.ranker.rank(query=question, docs=context, doc_ids=[i for i in range(len(context))])
        elif algo_type == 'mod_hyde':
            mod_hyde_answer = get_mod_HyDE_answer(question)
            context,metadata = self.retrieve(mod_hyde_answer)
            if self.use_reranker: results = self.ranker.rank(query=mod_hyde_answer, docs=context, doc_ids=[i for i in range(len(context))])
        if self.use_reranker:
            rerank_context = []
            rerank_ids = []
            for idx,res in enumerate(results.results):
                if idx+1 == self.rerank_docs:
                    break
                else:
                    rerank_context.append(res.text)
                    rerank_ids.append(res.doc_id)
            prediction = self.generate_answer(context=rerank_context, question=question)
            print("P: ",prediction)
            return dspy.Prediction(answer=prediction.answer,metadata=[metadata[rerank_id] for rerank_id in rerank_ids],context=rerank_context)
        else:
            prediction = self.generate_answer(context=context[:self.rerank_docs], question=question)
            
            return dspy.Prediction(answer=prediction.answer,reasoning=prediction.reasoning,metadata=metadata[:self.rerank_docs],context=context)