from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import  PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from dotenv import load_dotenv
import os

# load_dotenv(dotenv_path=r'C:\Users\asoha\Desktop\cse\AI\.env')

# api_key = os.getenv('GEMINI_API_KEY')
# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

loader = DirectoryLoader(
    path='Langchain_Doc_Loaders',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load()

print(len(docs))    # combines all the pages of all the pdfs in the given path
