import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "sequential_app"
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

model = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'Genrate a detailed report on {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Genrate a 5 point summary from the following text \n {text}',
    input_variables = ['text']
)

chain = prompt1 | model | parser | prompt2 | model | parser

config = {'run_name':'sequence_chain',
          'tags':['groq_model'],
          'metadata':{'model':'llama-3.3-70b'}
         }

result = chain.invoke({"topic":"langchain"}, config=config)
print(result)