from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import  PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=r'C:\Users\asoha\Desktop\cse\AI\.env')

api_key = os.getenv('GEMINI_API_KEY')
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

loader = PyPDFLoader(r'Langchain_Doc_Loaders\Pdf.pdf')
docs = loader.load()
# len(docs) = no of pages in pdf    

# parser = StrOutputParser()

# template = PromptTemplate(
#     template="Write a 5 summary on the given text below\n{text}",
#     input_variables=['text']
# )

# chain = template | model | parser

# response = chain.invoke({
#     'text':docs[0].page_content
# })

# print(response)

print(len(docs))
print(docs[0].metadata)
print(docs[0].page_content)
print(docs[1].metadata)
print(docs[1].page_content)