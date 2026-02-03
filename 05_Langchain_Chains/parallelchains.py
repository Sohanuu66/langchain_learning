from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import  PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
import os

load_dotenv()

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

template3 = PromptTemplate(
    template="combine both the notes and quiz into one and format them well. Make sure that the questions are relavant to the notes generated. if not, make new questions as an alternative for those.\nNotes on the topic --> {notes}\nQuiz on the topic --> {quiz}",
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : template1 | model | parser,
    'quiz' : template2 | model | parser
})

end_chain = template3 | model | parser

chain = parallel_chain | end_chain
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


