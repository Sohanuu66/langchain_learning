from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=r'C:\Users\asoha\Desktop\cse\AI\.env')
api_key = os.getenv('GEMINI_API_KEY')

docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

vector_store = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key),
    persist_directory=r'Langchain_Retrievers\vector_db',
    collection_name='CCR'
)

vector_store.add_documents(docs)

retriever = vector_store.as_retriever(
    search_type='mmr',
    search_kwargs={'k':2, 'lambda_mult':0.8}
)

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=api_key)
compress = LLMChainExtractor.from_llm(model)

ccr_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=compress
)

result = ccr_retriever.invoke("What is photosynthesis?")

print(result)   