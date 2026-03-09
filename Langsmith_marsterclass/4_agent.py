import os
from langchain_groq import ChatGroq
from langchain_core.tools import tool
import requests
#from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_classic.agents import AgentExecutor,create_react_agent
from dotenv import load_dotenv
from langchain import hub

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Agent_chatbot"
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')  # Add to .env

search_tool = TavilySearchResults(max_results=3)


@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=99c30bec5c73e28b1c97a6ba959d5725&query={city}'

  response = requests.get(url)

  return response.json()


llm = ChatGroq(model="llama-3.3-70b-versatile")

# Step 2: Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt

# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

# What is the release date of Dhadak 2?
# What is the current temp of gurgaon
# Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.

# Step 5: Invoke
response = agent_executor.invoke({"input": "resurch in linkdin the person yogesh patwal in company euclid infotech and tell what is qualification and do work"})
print(response)

print(response['output'])

# from langchain_community.tools import DuckDuckGoSearchRun

# search = DuckDuckGoSearchRun()
# result = search.run("Dhadak 2 latest news")
# print(result)

