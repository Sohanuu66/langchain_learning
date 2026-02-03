from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=r'.env')
api_key = os.getenv('GEMINI_API_KEY')

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=api_key)

history = [
    SystemMessage(content='You are a helpful AI assistant')
]

while(True):
    user = input("You: ")
    if user == 'exit':
        break
    history.append(HumanMessage(content=user))
    response = model.invoke(history)
    history.append(AIMessage(content=response.content))
    print("AI :", response.content)

print(history)
