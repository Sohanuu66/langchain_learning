from langchain_experimental.text_splitters import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv(r"C:\Users\asoha\Desktop\cse\AI\.env")

text = """
Artificial Intelligence (AI) enables computers to perform tasks that typically require human intelligence.
Machine learning (ML) is a subset of AI that allows systems to learn from data.
Deep learning is a further subset of ML, which uses neural networks.
AI is used in healthcare, education, and finance to improve efficiency.
However, it raises concerns about privacy, ethics, and job displacement.
"""

# Create a semantic chunker
chunker = SemanticChunker(GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY")))

chunks = chunker.split_text(text)

for i, c in enumerate(chunks):
    print(f"\n--- Semantic Chunk {i+1} ---\n{c}")
