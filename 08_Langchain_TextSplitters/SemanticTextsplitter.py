from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

text = """
The history of artificial intelligence began in the 1950s when Alan Turing proposed the famous Turing Test. This test was designed to determine whether a machine could exhibit intelligent behavior indistinguishable from a human. Early AI researchers were incredibly optimistic, predicting that human-level AI would be achieved within a generation. However, the field soon encountered significant technical limitations and entered what became known as the first AI winter in the 1970s.

Machine learning emerged as a dominant paradigm in the 1980s and 1990s. Instead of programming explicit rules, researchers developed algorithms that could learn patterns from data. Neural networks, inspired by the structure of the human brain, gained popularity during this period. Deep learning, a subset of machine learning using multi-layered neural networks, revolutionized the field in the 2010s with breakthroughs in image recognition and natural language processing.

The practical applications of AI are vast and growing rapidly. In healthcare, AI systems can analyze medical images to detect diseases like cancer with accuracy comparable to expert radiologists. Machine learning algorithms predict patient outcomes and suggest personalized treatment plans. Drug discovery has been accelerated by AI models that can simulate molecular interactions and identify promising compounds in a fraction of the time traditional methods require.

The financial sector has embraced AI for fraud detection, algorithmic trading, and risk assessment. Banks use machine learning to identify unusual transaction patterns that might indicate fraudulent activity. Robo-advisors provide automated investment advice based on individual financial goals and risk tolerance. Credit scoring models have become more sophisticated, incorporating non-traditional data sources to assess creditworthiness.

Transportation is being transformed by autonomous vehicle technology. Self-driving cars use computer vision, sensor fusion, and deep learning to navigate roads safely. Companies like Tesla, Waymo, and Cruise are testing and deploying autonomous vehicles in various cities. The potential benefits include reduced traffic accidents, increased mobility for elderly and disabled individuals, and more efficient use of road infrastructure.

However, the rise of AI brings significant ethical challenges and societal concerns. Privacy is a major issue as AI systems often require vast amounts of personal data to function effectively. There are concerns about algorithmic bias, where AI systems may perpetuate or amplify existing societal prejudices present in training data. Facial recognition technology raises questions about surveillance and civil liberties.

Job displacement is another critical concern as AI automation advances. While new jobs will be created, many traditional roles may become obsolete. Workers in transportation, manufacturing, and customer service face particular risk. Society must grapple with questions about retraining programs, social safety nets, and the distribution of wealth in an increasingly automated economy.

The future of AI remains uncertain but full of possibilities. Some researchers work toward artificial general intelligence (AGI), a system with human-level reasoning across all domains. Others focus on narrow AI applications that excel at specific tasks. Questions about AI consciousness, rights, and alignment with human values become more pressing as systems grow more sophisticated. International cooperation on AI governance and safety will be essential to ensure that this powerful technology benefits all of humanity.
"""

# Use HuggingFace Inference API (requires API token)
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

chunker = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile", # 'standard_deviation'
    breakpoint_threshold_amount=80,
    number_of_chunks=5
)

chunks = chunker.split_text(text)

for i, c in enumerate(chunks):
    print(f"\n--- Semantic Chunk {i+1} ---\n{c}")