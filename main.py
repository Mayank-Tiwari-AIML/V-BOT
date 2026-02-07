from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")

app = FastAPI(title="VIT BOT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_PATH = "chroma_db"


def get_rag_chain():
    # Embeddings using environment key
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    # Load existing database
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    prompt = PromptTemplate(
        template="""You are a helpful assistant. Answer ONLY from the provided context, if you do not find answer just reply (Information not found in database , check if there is typo)
Context: {context}
Question: {question}""",
        input_variables=["context", "question"]
    )

    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(
                lambda docs: "\n\n".join(d.page_content for d in docs)
            ),
            "question": RunnablePassthrough(),
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


class QuestionRequest(BaseModel):
    question: str


# Initialize chain once at startup (faster)
rag_chain = get_rag_chain()


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        answer = rag_chain.invoke(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
