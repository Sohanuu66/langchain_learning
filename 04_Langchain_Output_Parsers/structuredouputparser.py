from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Annotated
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=api_key)

class Str(BaseModel):
    name : Annotated[str, "name of the fictional character"]
    city : Annotated[str, "city name from which the ficitional character came from and it should be 1 word"]
    age : Annotated[int, Field(ge=16), "age of the fictional character. Incase of many ages in different versions, pick the most common/ famous one."]

parser = PydanticOutputParser(pydantic_object=Str)


template = PromptTemplate(
    template='Generate the name, city and age of a fictional person from the movie {movie}\n {format_instruction}',
    input_variables=['movie'],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

# prompt = template.invoke({
#     'movie' : 'Toy Story'
# })

# response = model.invoke(prompt)

# str_ouput = parser.parse(response.content)

chain = template | model | parser

str_ouput = chain.invoke({
    'movie' : 'Terminator'
})

print(str_ouput)