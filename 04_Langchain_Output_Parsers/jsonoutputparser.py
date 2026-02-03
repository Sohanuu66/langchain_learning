from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=r'.env')

api_key = os.getenv('GEMINI_API_KEY')
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=api_key)

parser = JsonOutputParser()

template = PromptTemplate(
    template="give me name, age and gender of a fictional person\n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction' : parser.get_format_instructions()} # not for strouputparser    
)
#  Return a JSON object. --> get_format_instructions() basically gives this

# prompt = template.invoke({})

# response = model.invoke(prompt)

# parsed_response = parser.parse(response.content)

chain = template | model | parser
parsed_response = chain.invoke({})

print(parsed_response)
print(type(parsed_response))