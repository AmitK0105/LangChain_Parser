from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm= HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.1", task="text-generation", max_new_tokens=100)

#llm= HuggingFaceEndpoint(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation")

model= ChatHuggingFace(llm=llm)

# 1st prompt--> Detailed report

template1= PromptTemplate(template="write a detailed report on {topic}", input_variables=['topic'])

# 2nd Prompt

template2= PromptTemplate(template="write a 5 line summary on the following text. /n {text}",
                          input_variables=['text'])


parser= StrOutputParser()

chain= template1| model | parser | template2 | model | parser


result= chain.invoke({"topic": "what is black scholes model in credit risk framework ?"})

print(result)

