#Pydantic output parser

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
#from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm= HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.1", task="text-generation", max_new_tokens=100)

#llm= HuggingFaceEndpoint(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation")

model= ChatHuggingFace(llm=llm)

# Create the Pydantic class

class Person(BaseModel):
    Name: str= Field(description="Name of the person")
    City :str= Field(description="Name of the city the person belongs")
    Age :int= Field(gt=18, description= "Age of the person" )


# create a parser

parser= PydanticOutputParser(pydantic_object=Person)

# create a template

template= PromptTemplate(template="generate the Name, City and Age  of the fictional {place} person \n {format_instruction} ",
                         input_variables=["place"],
                         partial_variables={"format_instruction":parser.get_format_instructions()})

# create a prompt

prompt= template.invoke({"place":"India"})

# invoke the model

result= model.invoke(prompt)

# Send theis result to parser

final_result= parser.parse(result.content)

# print the final output

print(final_result)