from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
# model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)

# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)

# llm = HuggingFacePipeline(pipeline=pipe)
# chat_model = ChatHuggingFace(llm=llm)

# response = chat_model.invoke("What is the capital of India?") 
# print(response.content)

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs = dict(
        temperature = 0,
        max_new_tokens = 100
    )
)

chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke("What is the capital of India?") 
print(response.content)
