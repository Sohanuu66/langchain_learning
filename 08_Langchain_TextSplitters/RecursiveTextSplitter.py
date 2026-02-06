from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import  PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, Language
from dotenv import load_dotenv
import os

# load_dotenv(dotenv_path=r'C:\Users\asoha\Desktop\cse\AI\.env')

# api_key = os.getenv('GEMINI_API_KEY')
# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

loader = DirectoryLoader(
    path='07_Langchain_Doc_Loaders',
    glob='text.txt',
    loader_cls=TextLoader
)

docs = loader.load()

# splitter = CharacterTextSplitter(
#     chunk_size = 100,
#     chunk_overlap = 0,
#     separator=''
# )

text = """
# Artificial Intelligence (AI)

Artificial Intelligence (AI) is the simulation of human intelligence in machines that are designed to think, learn, and act like humans. It enables systems to **analyze data**, **learn from experience**, and **make decisions** with minimal human intervention.

---

## 🧠 Types of AI

1. **Narrow AI** – Designed for a specific task (e.g., voice assistants like Siri).
2. **General AI** – Has human-like intelligence across multiple domains.
3. **Superintelligent AI** – Hypothetical AI that surpasses human intelligence.

---

## 📚 Applications

| Sector        | Use Case                            | Example               |
|----------------|-------------------------------------|-----------------------|
| Healthcare     | Disease detection                   | Radiology AI systems  |
| Education      | Personalized learning               | Smart tutoring bots   |
| Finance        | Fraud detection                     | Transaction scanners  |
| Transportation | Autonomous vehicles                 | Self-driving cars     |

---

## ⚙️ Machine Learning Example

Below is a simple example of linear regression implemented in Python:

```python
from sklearn.linear_model import LinearRegression

# Training data
X = [[1], [2], [3], [4]]
y = [2, 4, 6, 8]

model = LinearRegression()
model.fit(X, y)

print(model.predict([[5]]))  # Output: [10.]

"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language= Language.MARKDOWN,
    chunk_size = 300,
    chunk_overlap = 0
)

# result = splitter.split_documents(docs)
result = splitter.split_text(text)
print(result)
print(len(result))    # combines all the pages of all the pdfs in the given path