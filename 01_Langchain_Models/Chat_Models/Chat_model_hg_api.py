from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# HuggingFaceEndpoint is a LangChain LLM wrapper that allows you to connect to the Hugging Face Inference API 
# (hosted models on Hugging Face Hub).

load_dotenv()

api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')

llm  = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=api_key
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")

print(result.content)