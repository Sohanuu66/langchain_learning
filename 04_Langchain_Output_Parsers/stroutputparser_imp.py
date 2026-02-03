from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# prompt --> LLM --> response + prompt --> LLM --> final response

load_dotenv(dotenv_path=r'.env')
api_key = os.getenv('GEMINI_API_KEY')

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=api_key)

template1 = PromptTemplate(
    template="Give me a detailed description on the topic : {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Give me a 5 lines of summary on the following: \n{text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain  = template1 | model | parser | template2 | model | parser
# safer chain is
# description_chain = template1 | model | parser
# summary_chain = template
response = chain.invoke({
    'topic' : 'Langchain'
})

print(response)
