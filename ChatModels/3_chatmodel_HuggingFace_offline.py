from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# Load a local model
hf_pipeline = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Wrap it in LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Generate text
response = llm.invoke("what is capital of India ?")
print(response)