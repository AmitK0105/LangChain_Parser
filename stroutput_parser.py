#String output parser

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm= HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.1", task="text-generation", max_new_tokens=100)

#llm= HuggingFaceEndpoint(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation")

model= ChatHuggingFace(llm=llm)

# 1st prompt--> Detailed report

template1= PromptTemplate(template="write a detailed report on {topic}", input_variables=['topic'])

# 2nd Prompt

template2= PromptTemplate(template="write a 5 line summary on the following text. /n {text}",
                          input_variables=['text'])


prompt1= template1.invoke({"topic":"Black scholes models in Credit risk framework"})

result= model.invoke(prompt1)

prompt2= template2.invoke({"text": result.content})

result2= model.invoke(prompt2)

print(result2.content)