from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(dotenv_path=r'.env')

api_key = os.getenv('GEMINI_API_KEY')
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=api_key)

# Define Pydantic schema
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

# Create parser
parser = JsonOutputParser(pydantic_object=Joke)

# Create prompt
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Build chain
chain = prompt | model | parser

# Run chain
result = chain.invoke({"query": "Tell me a joke."})

# Access structured output
print("\n✅ Structured Output (Pydantic Object):", result)
# returns a dict
