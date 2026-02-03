from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')

template1 = PromptTemplate(
    template="Explain in brief about this topic {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Give a 5 pointer summary about this text : {text}",
    input_variables=['text']
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

response = chain.invoke({
    'topic' : 'Agent Clawdbot'
})

print(response)

chain.get_graph().print_ascii()

