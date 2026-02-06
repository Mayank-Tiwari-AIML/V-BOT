from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")

print("Import complete")

# FastAPI app

app = FastAPI(title="VIT BOT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all domains (for dev)
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load MULTIPLE CSV files

csv_files = [
    "data/cleaned_website_data.csv",
    "data/academic_calendar_detailed.csv",
    "data/data1.csv",
    "data/ratings_final.csv"   # second file
]

print("loaded")

documents = []

for file_path in csv_files:
    loader = CSVLoader(file_path=file_path, encoding="utf-8")
    documents.extend(loader.load())

print("loader done")

# Split documents

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)
print("splitted")

# Embeddings & Vector Store

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

print("embedded")

vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "lambda_mult": 0.4}
)

print("retrival done")

# LLM

llm = ChatOpenAI(
    model="gpt-oss-120b",
    temperature=0.2,
    openai_api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

print("llm initialised")

# Prompt

prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided context.
If the context is insufficient, give an answer you know , 
then also you are unable to find say,"infomation not found in database"

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

print("promp generated")

# Chain

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)

print("chain generated")

# Request Schema

class QuestionRequest(BaseModel):
    question: str

print("completed")

# API Endpoint

@app.post("/ask")
def ask_question(request: QuestionRequest):
    answer = rag_chain.invoke(request.question)
    return {
        "question": request.question,
        "answer": answer
    }
