from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

docs = [
    'Virat Kohli is the best cricketer in the world',
    'Sachin tendulkar is called the god of cricket',
    'Dhoni is called the captain cool',
    'Jasprit bumrah is the best bowler India has ever seen'
]

embedding = OpenAIEmbeddings(model='text-embedding-3-large' , dimensions=32)

result = embedding.embed_documents(docs)

query = "Who is Virat Kohli ?" 

q_embedding = embedding.embed_query(query)

scores = cosine_similarity([q_embedding],result)[0]
index, similairty_score = (sorted(list(enumerate(scores)), key=lambda x:x[1])[-1])

print(query)
print(docs[index])

print('Similarity Score is : ' + str(similairty_score))