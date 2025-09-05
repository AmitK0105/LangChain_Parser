# json output parser

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm= HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.1", task="text-generation", max_new_tokens=100)

#llm= HuggingFaceEndpoint(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation")

model= ChatHuggingFace(llm=llm)

parser= JsonOutputParser()

template1= PromptTemplate(template="Please write a detailed summary of black holes. \n {format_instruction}"
                           ,input_variables=[],
                          partial_variables={'format_instruction':parser.get_format_instructions()}
                          )

# prompt= template1.format()

# #print(prompt)

# result= model.invoke(prompt)

# #print(result)

# final_result= parser.parse(result.content)

# print(final_result['keyComponents'])
# print(type(final_result))

# through the use of chain

chain= template1 | model | parser

result= chain.invoke({})

print(result)