from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory
from dotenv import load_dotenv
import os

def retrieval(vector_store: PineconeVectorStore,question, knn, ):

    query = question
    Generalcontext = vector_store.similarity_search_with_score(query, k = knn, namespace = "General")
    Coursecontext = vector_store.similarity_search_with_score(query, k = knn, namespace = "Courses")

    allcontext = Generalcontext + Coursecontext

    allcontext = sorted(allcontext, key = lambda x: x[1], reverse=True)

    if (len(allcontext) >= knn):
        context = [allcontext[i][0] for i in range(knn)]
    else:
        context = [allcontext[i][0] for i in range(len(allcontext))]

    return context

def createprompt():
    temp = "please answer the following question {question} using the context {context}. You are an assistant in Queen mary's university of london and you are talking to a student, if there are no documents provided then prompt user to elaborate on their question. please use a friendly demeanor. write 2 paragraphs max"
    prompt = ChatPromptTemplate.from_template(temp)
    return prompt


def generation(vector_store: PineconeVectorStore,question,  model : ChatOpenAI, context, prompt):
    chain = prompt | model | StrOutputParser()

    b = chain.invoke({"context":context, "question":question})

    return b

def full_pipeline(vectorstore: PineconeVectorStore,question, knn, model : ChatOpenAI):
    context = retrieval(vector_store=vectorstore, question=question,knn = knn)
    prompt = createprompt()
    response = generation(vector_store=vectorstore,question=question,model=model,context=context, prompt=prompt)
    return response