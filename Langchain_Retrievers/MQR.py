from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=r'C:\Users\asoha\Desktop\cse\AI\.env')
api_key = os.getenv('GEMINI_API_KEY')

api_key = os.getenv('GEMINI_API_KEY')  

docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

vector_store = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key),
    persist_directory=r'Langchain_Retrievers\vector_db',
    collection_name='Health'
)

vector_store.add_documents(docs)

multiquery_prompt = PromptTemplate.from_template(
    """
You are a helpful assistant that reformulates user queries into multiple related search queries
to improve retrieval coverage.

Generate 5 different versions of the following query, 
each phrased uniquely but with the same meaning or related focus.

User Query: {question}

Return each reformulated query on a new line.
    """
)

retriever = vector_store.as_retriever(
    search_type='mmr',
    search_kwargs={'k':5, 'lambda_mult':0.8}
)
# 1-behaves like similarity search
# 0-highly diverse
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key),
    prompt=multiquery_prompt
)

query = "how to improve energy levels and maintain balance"

result1 = multiquery_retriever.invoke(query)
result2 = retriever.invoke(query)

print(result1)
print()
print(result2)