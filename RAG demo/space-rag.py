from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI



load_dotenv()

# Load the PDF file
loader = PyPDFLoader("/Users/shivendragupta/Desktop/Langchain Models/RAG demo/06. Astronomy An overview Autor Poincare.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)

docs = text_splitter.split_documents(documents)

vector_store = FAISS.from_documents(docs , OpenAIEmbeddings())

retreiver = vector_store.as_retriever()

query = "What is infrared astronomy ?"

retrived_docs = retreiver.get_relevant_documents(query)

retre = '\n'.join([doc.page_content for doc in retrived_docs])

print(retre)

# llm = OpenAI(model='gpt-4')

prompt = f"Based on following text , answer the question : {query} \n\n {retre}"

# res = llm.invoke(prompt)
# print(res.content)

model = ChatOpenAI(model='gpt-4',temperature=1.5) # Randomness temperature

result = model.invoke(prompt)

print(result.content)