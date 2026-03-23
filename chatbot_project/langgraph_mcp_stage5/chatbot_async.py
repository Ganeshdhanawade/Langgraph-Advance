from langgraph.graph import StateGraph, START, END
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import asyncio

load_dotenv()

# ------------------
# 1.LLM
# ------------------
llm = ChatGroq(model="llama-3.1-8b-instant",temperature=0) #llama-3.3-70b-versatile

# ------------------
# 2. Tool
# ------------------

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
    

tools = [calculator]
llm_with_tools = llm.bind_tools(tools)

# ------------------
# 3. State
# ------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

def build_graph():
    
    async def chat_node(state:ChatState):
        """ LLM node that may answer or request a tool call."""
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages":[response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(ChatState)
    graph.add_node("chat_node",chat_node)
    graph.add_node("tools",tool_node)

    graph.add_edge(START,"chat_node")

    graph.add_conditional_edges("chat_node",tools_condition)
    graph.add_edge('tools','chat_node')

    chatbot = graph.compile()

    return chatbot


async def main():

    chatbot = build_graph()

    #ruuing the graph
    result = await chatbot.ainvoke({'messages': [HumanMessage(content="find the modulus of 12345 and 23 and give answer like acricket commentator.")]})

    print(result['messages'][-1].content)

if __name__=="__main__":
    asyncio.run(main())