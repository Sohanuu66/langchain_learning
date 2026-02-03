from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import  PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, WebBaseLoader
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=r'C:\Users\asoha\Desktop\cse\AI\.env')

api_key = os.getenv('GEMINI_API_KEY')
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

loader = WebBaseLoader('https://www.amazon.in/s?k=ashwagandha&crid=14SUGG9C59RRQ&sprefix=ashwagandha%2Caps%2C261&ref=nb_sb_noss_2')

docs = loader.load()

template = PromptTemplate(
    template="Give me top 5 best ashwagandha out of all on the basis of price and rating?\n{text}",
    input_variables=['text']
)
parser = StrOutputParser()

chain = template | model | parser

response = chain.invoke({
    'text' : docs[0].page_content
})

print(response)    # combines all the pages of all the pdfs in the given path
