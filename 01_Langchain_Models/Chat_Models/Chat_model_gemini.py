from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature = 0)

# Simple text invocation
result = llm.invoke("What is the capital of India?")
print(result.content)

# Multimodal invocation with gemini-pro-vision
# message = HumanMessage(
#     content=[
#         {
#             "type": "text",
#             "text": "What's in this image?",
#         },
#         {"type": "image_url", "image_url": "https://picsum.photos/seed/picsum/200/300"},
#     ]
# )
# result = llm.invoke([message])
# print(result.content)