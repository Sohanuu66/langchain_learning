from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=r'.env')

api_key = os.getenv('GEMINI_API_KEY')
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=api_key)

parser = StrOutputParser()

template = PromptTemplate(
    template="give me name, age and gender of a fictional person\n {char}",
    input_variables=['char'],    
)

prompt = template.invoke({'char':'spider man'})

response = model.invoke(prompt)

parsed_response = parser.invoke(response)
print(parsed_response)