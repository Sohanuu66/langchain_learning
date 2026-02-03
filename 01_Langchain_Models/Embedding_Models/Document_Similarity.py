from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

query = "tell me about bumrah"

docs_embeds = embeddings.embed_documents(documents)
query_embeds = embeddings.embed_query(query)

scores = cosine_similarity([query_embeds], docs_embeds)[0]
index, score = sorted(list(enumerate(scores)), key = lambda x : x[1])[-1]

print(query)
print(documents[index])
print("Similarity score: ", score)



