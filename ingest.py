import os
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# Configuration
CHROMA_PATH = "chroma_db"
CSV_FILES = [
    "data/cleaned_website_data.csv",
    "data/academic_calendar_detailed.csv",
    "data/data1.csv",
    "data/ratings_final.csv"
]

def create_db():
    # 1. Load Documents
    documents = []
    for file_path in CSV_FILES:
        if os.path.exists(file_path):
            loader = CSVLoader(file_path=file_path, encoding="utf-8")
            documents.extend(loader.load())
    
    # 2. Split Text
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # 3. Create Embeddings & Save to Disk
    # Note: Use your OpenRouter key here just for the setup phase
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH
    )
    print(f"Database created at {CHROMA_PATH}")

if __name__ == "__main__":
    create_db()