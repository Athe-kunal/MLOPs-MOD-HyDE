# MLOPs-MOD-HyDE

This project is a work in progress based on our [hackathon](https://github.com/Athe-kunal/AD-Finance-Agent) on a expert driven financial chatbot. Let's discuss the MLOPs system design of the project

## THE "WHAT" AND "WHY"

### Background
This system will help users to do public and private company valuations based on expert-driven approach. The Signal to Noise ratio in the financial world is really low and we generaly don't have expert advice during valuations. This system will solve the problem with parsing through the overflowing information. It will help them to plan on how to do valuations, the mathematical concepts involved in valuations. It will be similar to a professor teaching you about valuation.


### Value Proposition
The product is a question answering agent that can refer to the provided documents from books and YouTube recordings to answer questions regarding financial evaluations. The answers will be grouned in sources from books and YouTube recordings, so that ysers can verify the answers and read more about it. It alleviates the pain points of going through all the textbooks and YouTube videos, also answers are grounded based on expert-driven approach.

### Objectives
* We have to convert the pdfs to text using unstructured or convert the pdfs to markdowns using marker-pdf. This will help us in ingesting our data.
* Also, we are using a novel technique called as Modified HyDE, where like HyDE first we finetune a small causal language model on our raw dataset to make it a glorified next token predictor. It will help us rewriting the query, so that we can use it in Retrieval augmented generation (RAG). This is better than HyDE, where we call a general purpose model
* Then we will build a vector database to store the embeddings of our data. During query time, our question will pass through the finetuned causal language model first, then we will pass to our RAG system to get answers. The modified HyDE and HyDE closes the distribution gap between the question and the data source by first hallucinating that question and then retrieving from the data source.  