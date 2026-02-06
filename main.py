from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="VIT BOT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_PATH = "chroma_db"

def get_rag_chain(api_key: str):
    # Initialize embeddings with user's key
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Load existing database from disk
    vector_store = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    prompt = PromptTemplate(
        template="""You are a helpful assistant. Answer ONLY from the provided context.
Context: {context}
Question: {question}""",
        input_variables=["context", "question"]
    )

    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough(),
        })
        | prompt | llm | StrOutputParser()
    )
    return chain

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QuestionRequest, x_api_key: Optional[str] = Header(None)):
    if not x_api_key:
        raise HTTPException(status_code=400, detail="API Key is missing")
    
    try:
        chain = get_rag_chain(x_api_key)
        answer = chain.invoke(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))