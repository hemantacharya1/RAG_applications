from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import os

def respond_to_query(query: str):
    """
    Responds to a user query by retrieving relevant information from the
    FAISS vector store and generating a response using an LLM.
    """
    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    llm_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    if not os.path.exists("faiss_index"):
        return "Vector store not found. Please ingest documents first."

    vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

    docs = vectorstore.similarity_search(query)
    
    if not docs:
        return "No relevant documents found for your query."

    prompt_template = """
    You are a helpful AI assistant. Answer the user's question based on the provided context.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Provide the answer clearly and concisely.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = load_qa_chain(llm_model, chain_type="stuff", prompt=prompt)
    
    response = chain.run(input_documents=docs, question=query)

    return response