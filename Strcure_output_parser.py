# Structure output parser- It allow us to create the schema
# using chain

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
#from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm= HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.1", task="text-generation", max_new_tokens=100)

#llm= HuggingFaceEndpoint(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation")

model= ChatHuggingFace(llm=llm)

schema= [ResponseSchema(name= "fact1", description="fact1 about the topic"),
         ResponseSchema(name= "fact2", description="fact2 about the topic"),
         ResponseSchema(name= "fact3", description="fact3 about the topic")
         ]

parser= StructuredOutputParser.from_response_schemas(schema)

template1= PromptTemplate(template="Give 3 facts about the {topic} \n {format_instruction}",
                          input_variables=['topic'],
                          partial_variables={"format_instruction": parser.get_format_instructions()}
                          )

#prompt= template1.invoke({"topic":"black sholes model"})

chain= template1 | model | parser

#result= model.invoke(prompt)

#final_result= parser.parse(result.content)

final_result= chain.invoke({"topic":"black scholes model"} )

print(final_result)