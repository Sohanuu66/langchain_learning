from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=r'.env')
api_key = os.getenv('GEMINI_API_KEY')

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=api_key)

template = ChatPromptTemplate(
    [
        ("system", "You are a helpful AI bot."),
        ("placeholder", "{conversation}"),  
    ]
)

conversation = [
            ("human", "Hi!"),
            ("ai", "How can I assist you today?"),
            ("human", "Which number is greater, 2 or 0"),
            ("ai", "2 is greater"),
        ]

while(True):
    user = input("You: ")
    if user == 'exit':
        break
    conversation.append(("human", user))
    prompt = template.invoke(
        {
            "conversation" : conversation
        }
    )
    response = model.invoke(prompt)
    conversation.append(("ai", response.content))
    print(response.content)

prompt = template.invoke(
        {
            "conversation" : conversation
        }
    )
print(prompt)

# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# # chat template
# chat_template = ChatPromptTemplate([
#     ('system','You are a helpful customer support agent'),
#     MessagesPlaceholder(variable_name='chat_history'),
#     ('human','{query}')
# ])

# chat_history = []
# # load chat history
# with open('chat_history.txt') as f:
#     chat_history.extend(f.readlines())

# print(chat_history)

# # create prompt
# prompt = chat_template.invoke({'chat_history':chat_history, 'query':'Where is my refund'})

# print(prompt)