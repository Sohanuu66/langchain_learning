from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate

# prompt --> LLM --> response + prompt --> LLM --> final response

load_dotenv()
api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')

llm = HuggingFaceEndpoint(
    repo_id='HuggingFaceH4/zephyr-7b-beta',
    huggingfacehub_api_token=api_key,
    task='text-generation'
)

model = ChatHuggingFace(llm= llm)

template1 = PromptTemplate(
    template="Give me a detailed description on the topic : {topic}",
    input_variables=['topic']
)

prompt1 = template1.invoke({
    'topic' : 'langchain'
})

result1 = model.invoke(prompt1).content

print(result1)

template2 = PromptTemplate(
    template="Give me a 5 lines of summary on the following: \n{text}",
    input_variables=['text']
)

prompt2 = template2.invoke({
    'text' : result1
})

result2 = model.invoke(prompt2).content

print(result2)
