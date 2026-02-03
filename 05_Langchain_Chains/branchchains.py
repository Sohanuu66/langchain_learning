from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import  PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

class Review(BaseModel):
    sentiment : Annotated[Literal['Positive', 'Negative'], Field(description="Give the sentiment of the given review")]
    topic : Annotated[str, Field(description='Give a one to two words on which topic the review was all about')]

parser = PydanticOutputParser(pydantic_object=Review)

parser2 = StrOutputParser()

template1 = PromptTemplate(
    template='Give the sentiment and one to two words topic name of the following review\n{review}\n{format_instruction}',
    input_variables=['review'],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

base_chain = template1 | model | parser

template2 = PromptTemplate(
    template='Generate a response (only 1) over a positive sentiment review given by the customer on the topic :{topic}',
    input_variables=['topic']
)

template3 = PromptTemplate(
    template='Generate a response (only 1) over a negative sentiment review given by the customer on the topic :{topic}',
    input_variables=['topic']
)

pos_chain = (
    RunnableLambda(lambda x:{'topic' : x.topic}) | template2 | model | parser2
)

neg_chain = (
    RunnableLambda(lambda x: {'topic' : x.topic}) | template3 | model | parser2
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'Positive', pos_chain),    # cond1
    (lambda x:x.sentiment == 'Negative', neg_chain),    # cond2
    RunnableLambda(lambda x : "invalid sentiment")      # default
)

# result = base_chain.invoke({
#     'review' : "This is a really good laptop"
# })

chain = base_chain | branch_chain

result = chain.invoke({
    'review' : "This is a really good laptop"
})

print(result)

base_chain.get_graph().print_ascii()

