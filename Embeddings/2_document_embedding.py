from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large' , dimensions=32)

doc = [
    'The capital of India is Delhi',
    'France is a friend',
    'A good man is a good man'
]

results = embedding.embed_documents(doc)

print(results)
