import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "langsmith-demo"
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

## simple line prompt
prompt = PromptTemplate.from_template("{question}")

model = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()

#chain
chain = prompt | model | parser

result = chain.invoke({"question":"what is the capital of peru?"})
print(result)