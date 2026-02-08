from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import WikipediaRetriever
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')   

query = "sachin tendulkar"

retriever = WikipediaRetriever(top_k_results=2, lang='en')

docs = retriever.invoke(query)

# print(docs)

for i, doc in enumerate(docs):
    print(f"--------result{i + 1}------------")
    print(doc.page_content)  