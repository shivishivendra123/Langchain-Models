from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()

# Load the PDF file
loader = PyPDFLoader("/Users/shivendragupta/Desktop/Langchain Models/RAG demo/06. Astronomy An overview Autor Poincare.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=50)

docs = text_splitter.split_documents(documents)

vector_store = FAISS.from_documents(docs , HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))

retreiver = vector_store.as_retriever()

query = "What is Leave policy in days?"

retrived_docs = retreiver.get_relevant_documents(query)

retre = '\n'.join([doc.page_content for doc in retrived_docs])

# test = "Leave policy of XYZ in 500000 days"

prompt = f"Based on following text , answer the question : {query} \n\n {test}"

print(prompt)

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs = dict(
        temperature = 1,
        max_new_token = 200
    )
)

chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke(prompt) 
print(response.content)