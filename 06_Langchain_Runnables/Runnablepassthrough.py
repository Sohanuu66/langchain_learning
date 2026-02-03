from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import  PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=r'C:\Users\asoha\Desktop\cse\AI\.env')

api_key = os.getenv('GEMINI_API_KEY')
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

template1 = PromptTemplate(
    template="Generate short, crisp and exam friendly notes on the given text below\n{text}",
    input_variables=['text']
)

template2 = PromptTemplate(
    template="Generate 5 exam worthy and frequently asked questions and answers on the given text below\n{text}",
    input_variables=['text']
)

parser = StrOutputParser()

start_chain = template1 | model | parser

parallel_chain = RunnableParallel({
    'quiz' : template2 | model | parser,
    'notes' : RunnablePassthrough()
})

chain = start_chain | parallel_chain

result = chain.invoke({
    'text' : """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and act like humans. 
AI systems can perform tasks such as learning, reasoning, problem-solving, perception, and language understanding. 
There are two main types of AI: narrow AI, which is designed for specific tasks like voice assistants or image recognition, and general AI, which can perform any intellectual task a human can do. 
AI applications are widespread, including healthcare (disease detection), education (personalized learning), transportation (autonomous vehicles), and finance (fraud detection). 
While AI improves efficiency and accuracy, it also raises concerns about job displacement, privacy, and ethical use. 
Responsible AI development focuses on fairness, transparency, and accountability to ensure AI benefits society as a whole.
"""
})

print(result)

chain.get_graph().print_ascii()


