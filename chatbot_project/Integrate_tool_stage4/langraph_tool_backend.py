from langgraph.graph import StateGraph, START, END
from typing import TypedDict,Annotated
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
import sqlite3

import requests
import random

load_dotenv()

# ------------------
# 1.LLM
# ------------------
llm = ChatGroq(model="llama-3.1-8b-instant",temperature=0) #llama-3.3-70b-versatile

# ------------------
# 2. Tool
# ------------------
# a.tool
search_tool = TavilySearchResults(max_results=3)

# b.tool
@tool
def calculator(first_num:float,sencond_num:float,opration: str) -> dict:
    """
    Perform the basic arithmatic opration on two numbers.
    supported opration: add, sub, mul, div
    """
    try:
        if opration == "add":
            result = first_num + sencond_num
        elif opration == "sub":
            result = first_num - sencond_num
        elif opration == "mul":
            result = first_num * sencond_num
        elif opration == "div":
            if sencond_num==0:
                return {"error":"division by zero is not allowed"}
            result = first_num / sencond_num
        else:
            return {"error" : f"Unsupported opration '{opration}'"}
        
        return {"first_num":first_num,"sencond_num":sencond_num,"opration":opration,"result":result}
    
    except Exception as e:
        return {"error": str(e)}
    
# c.tool
@tool
def get_stock_price(symbol:str) -> dict:
    """
    fetch latest stock price of the given symbol (e.g. 'AAPL','TSLA')
    using Alpha Vantage with API key in the URL
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=ZGNXYJ4YFBRBG71F"
    r = requests.get(url)
    return r.json()


tools = [search_tool,get_stock_price,calculator]
llm_with_tools = llm.bind_tools(tools)

# ------------------
# 3. State
# ------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

# ------------------
# 4. Nodes
# ------------------
def chat_node(state:ChatState):
    """ LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages":[response]}

tool_node = ToolNode(tools)

# ------------------
# 5. checkpointer
# ------------------
conn = sqlite3.connect(database="chatbot.db",check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 6. Graph
# -------------------

graph = StateGraph(ChatState)
graph.add_node("chat_node",chat_node)
graph.add_node("tools",tool_node)

graph.add_edge(START,"chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools','chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
