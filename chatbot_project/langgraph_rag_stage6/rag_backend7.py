# from __future__ import annotations

# import os
# import sqlite3
# import tempfile
# from typing import Annotated, Any, Dict, Optional, TypedDict

# from dotenv import load_dotenv
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_community.vectorstores import FAISS
# from langchain_core.messages import BaseMessage, SystemMessage
# from langchain_core.tools import tool
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
# from langgraph.checkpoint.sqlite import SqliteSaver
# from langgraph.graph import START,StateGraph
# from langgraph.graph.message import add_messages
# from langgraph.prebuilt import ToolNode,tools_condition
# import requests

# load_dotenv()

# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# fastmcp_key = os.getenv("FASTMCP_API_KEY")

# # -------------------
# # 1. LLM + embeddings
# # -------------------
# llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0)#llama-3.1-8b-instant
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# # -------------------
# # 2. PDF retriever store (per thread)
# # -------------------
# _THREAD_RETRIEVERS : Dict[str,Any] = {}
# _THREAD_METADATA : dict[str,Any] = {}


# def _get_retriever(thread_id: Optional[str]):
#     """ Featch the retriever for a thread if avialable."""
#     if thread_id and thread_id in _THREAD_RETRIEVERS:
#         return _THREAD_RETRIEVERS[thread_id]
#     return None

# def ingest_pdf(file_bytes:bytes,thread_id:str, filename:Optional[str]=None) ->dict:
#     """
#     Build a FAISS retriver for the uploaded PDF and store it for the thread.
#     retrun a summary dict that can be surfaced in the UI.
#     """
#     if not file_bytes:
#         raise ValueError("No bytes received for ingestion.")
    
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#         temp_file.write(file_bytes)
#         temp_path =temp_file.name

#     try:
#         loader = PyMuPDFLoader(temp_path)
#         docs = loader.load()

#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size = 100, chunk_overlap=20,separators=["\n\n", "\n", " ", ""]
#         )
#         chunks = splitter.split_documents(docs)

#         vector_store = FAISS.from_documents(chunks, embeddings)
#         retriever = vector_store.as_retriever(
#             search_type="similarity",search_kwargs={"k":4}
#         )

#         _THREAD_RETRIEVERS[str(thread_id)] = retriever
#         _THREAD_METADATA[str(thread_id)] = {
#             "filename": filename or os.path.basename(temp_file),
#             "documents":len(docs),
#             "chunks":len(chunks),
#         }

#         return {
#             "filename": filename or os.path.basename(temp_file),
#             "documents":len(docs),
#             "chunks":len(chunks),
#         }
#     finally:
#         # the FAISS store keeps copies of the text, so the tiem file is safe to remove
#         try:
#             os.remove(temp_path)
#         except OSError:
#             pass
    

# # -------------------
# # 3. Tools
# # -------------------
# search_tool = TavilySearchResults(max_results=3)


# @tool
# def calculator(first_num: float, second_num: float, operation: str) -> dict:
#     """
#     Perform a basic arithmetic operation on two numbers.
#     Supported operations: add, sub, mul, div
#     """
#     try:
#         if operation == "add":
#             result = first_num + second_num
#         elif operation == "sub":
#             result = first_num - second_num
#         elif operation == "mul":
#             result = first_num * second_num
#         elif operation == "div":
#             if second_num == 0:
#                 return {"error": "Division by zero is not allowed"}
#             result = first_num / second_num
#         else:
#             return {"error": f"Unsupported operation '{operation}'"}

#         return {
#             "first_num": first_num,
#             "second_num": second_num,
#             "operation": operation,
#             "result": result,
#         }
#     except Exception as e:
#         return {"error": str(e)}


# @tool
# def get_stock_price(symbol:str) -> dict:
#     """
#     fetch latest stock price of the given symbol (e.g. 'AAPL','TSLA')
#     using Alpha Vantage with API key in the URL
#     """
#     url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=ZGNXYJ4YFBRBG71F"
#     r = requests.get(url)
#     return r.json()


# @tool
# def rag_tool(query: str, thread_id:Optional[str]=None) -> dict:
#     """
#     Retrieve relavant information from the uploaded PDF for this chat thread.
#     Always include the thread_id when calling this tool.
#     """
#     retriever = _get_retriever(thread_id)
#     if retriever is None:
#         return {
#             "error":"No document indexed for this chat. Upload a PDF first.",
#             "query":query,
#         }
    
#     result = retriever.invoke(query)
#     context = [doc.page_content for doc in result]
#     metadata = [doc.metadata for doc in result]

#     return {
#         "query":query,
#         "context":context,
#         "metadata":metadata,
#         "source_file":_THREAD_METADATA.get(str(thread_id),{}).get("filename"),
#     }

# tools = [search_tool,get_stock_price,calculator,rag_tool]
# llm_with_tool = llm.bind_tools(tools)


# # -------------------
# # 4. State
# # -------------------
# class ChatState(TypedDict):
#     messages: Annotated[list[BaseMessage], add_messages]

# # -------------------
# #5. Nodes
# #-------------------

# def chat_node(state: ChatState, config=None):
#     """LLM node that may answer or request a tool call."""
#     thread_id = None
#     if config and isinstance(config,dict):
#         thread_id = config.get("configurable",{}).get(thread_id)

#     system_message = SystemMessage(
#         content=(
#             "You are a helpful assistant. For questions about the uploaded PDF, call"
#             "the 'rag tool' and include the thread_id"
#             f"{thread_id}. You can also use web search, stock price, and"
#             "calculator tools when helpful. If no documnet is avialable ask user"
#             "to apload a PDF."
#         )
#     )
    
#     messages = [system_message, *state["messages"]]
#     response = llm_with_tool.invoke(messages,config=config)
#     return {"messages":[response]}

# tool_node =ToolNode(tools)

# # -------------------
# # 6. Checkpointer
# # -------------------
# conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
# checkpointer = SqliteSaver(conn=conn)

# # -------------------
# # 7. Graph
# # -------------------
# graph = StateGraph(ChatState)
# graph.add_node("chat_node", chat_node)
# graph.add_node("tools", tool_node)

# graph.add_edge(START, "chat_node")
# graph.add_conditional_edges("chat_node", tools_condition)
# graph.add_edge("tools", "chat_node")

# chatbot = graph.compile(checkpointer=checkpointer)

# # -------------------
# # 8. Helpers
# # -------------------
# def retrieve_all_threads():
#     all_threads = set()
#     for checkpoint in checkpointer.list(None):
#         all_threads.add(checkpoint.config["configurable"]["thread_id"])
#     return list(all_threads)


# def thread_has_document(thread_id: str) -> bool:
#     return str(thread_id) in _THREAD_RETRIEVERS


# def thread_document_metadata(thread_id: str) -> dict:
#     return _THREAD_METADATA.get(str(thread_id), {})


from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
import requests

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
fastmcp_key = os.getenv("FASTMCP_API_KEY")

# -------------------
# 1. LLM + embeddings
# -------------------
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, Any] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and str(thread_id) in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[str(thread_id)]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.
    Return a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyMuPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200, 
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
search_tool = TavilySearchResults(max_results=3)


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price of the given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=ZGNXYJ4YFBRBG71F"
    try:
        r = requests.get(url, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@tool
def rag_tool(query: str, thread_id: str) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    
    Args:
        query: The question to search for in the document
        thread_id: The current chat thread ID (automatically provided)
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
            "thread_id": thread_id,
        }
    
    try:
        result = retriever.invoke(query)
        context = [doc.page_content for doc in result]
        metadata = [doc.metadata for doc in result]

        return {
            "query": query,
            "context": context,
            "metadata": metadata,
            "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
            "thread_id": thread_id,
            "num_chunks": len(context)
        }
    except Exception as e:
        return {
            "error": f"Error retrieving from document: {str(e)}",
            "query": query,
            "thread_id": thread_id
        }


tools = [search_tool, get_stock_price, calculator, rag_tool]


# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Custom Tool Node
# -------------------
def custom_tool_node(state: ChatState, config=None):
    """
    Custom tool node that injects thread_id into rag_tool calls.
    
    ✅ CRITICAL: config=None (not config: dict) for LangGraph compatibility
    """
    if config is None:
        config = {}
    
    thread_id = config.get("configurable", {}).get("thread_id")
    last_message = state["messages"][-1]
    
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": []}
    
    tool_messages = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"].copy()
        
        # ✅ Inject thread_id for rag_tool
        if tool_name == "rag_tool":
            tool_args["thread_id"] = str(thread_id)
        
        # Find and execute the tool
        tool_to_call = next((t for t in tools if t.name == tool_name), None)
        
        if tool_to_call:
            try:
                result = tool_to_call.invoke(tool_args)
                tool_messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"],
                        name=tool_name,
                    )
                )
            except Exception as e:
                tool_messages.append(
                    ToolMessage(
                        content=f"Error executing {tool_name}: {str(e)}",
                        tool_call_id=tool_call["id"],
                        name=tool_name,
                    )
                )
        else:
            tool_messages.append(
                ToolMessage(
                    content=f"Tool '{tool_name}' not found",
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )
    
    return {"messages": tool_messages}


# -------------------
# 6. Chat Node
# -------------------
def chat_node(state: ChatState, config=None):
    """
    LLM node that may answer or request a tool call.
    
    ✅ CRITICAL: config=None (not config: dict) for LangGraph compatibility
    """
    if config is None:
        config = {}
    
    thread_id = config.get("configurable", {}).get("thread_id")
    
    # Check document status
    has_document = thread_id and str(thread_id) in _THREAD_RETRIEVERS
    doc_status = ""
    
    if has_document:
        doc_info = _THREAD_METADATA.get(str(thread_id), {})
        doc_status = (
            f"✅ Document available: {doc_info.get('filename')} "
            f"({doc_info.get('chunks')} chunks from {doc_info.get('documents')} pages)"
        )
    else:
        doc_status = "❌ No document indexed yet."
    
    system_message = SystemMessage(
        content=(
            f"You are a helpful assistant with multiple capabilities.\n\n"
            f"**Current Thread ID:** `{thread_id}`\n"
            f"**Document Status:** {doc_status}\n\n"
            f"**Available Tools:**\n"
            f"1. **rag_tool** - Search the uploaded PDF (requires thread_id='{thread_id}')\n"
            f"2. **search_tool** - Web search using Tavily\n"
            f"3. **get_stock_price** - Get stock prices (e.g., AAPL, TSLA)\n"
            f"4. **calculator** - Perform arithmetic operations\n\n"
            f"**Instructions:**\n"
            f"- For PDF questions: Use rag_tool with query and thread_id='{thread_id}'\n"
            f"- If no document is indexed, politely ask user to upload a PDF\n"
            f"- For other questions, use appropriate tools or answer directly\n"
        )
    )
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    
    return {"messages": [response]}


# -------------------
# 7. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


# -------------------
# 8. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", custom_tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# -------------------
# 9. Helper Functions
# -------------------
def retrieve_all_threads():
    """Retrieve all thread IDs from checkpoint history."""
    all_threads = set()
    try:
        for checkpoint in checkpointer.list(None):
            all_threads.add(checkpoint.config["configurable"]["thread_id"])
    except Exception as e:
        print(f"Error retrieving threads: {e}")
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    """Check if a thread has an indexed document."""
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    """Get metadata about the indexed document for a thread."""
    return _THREAD_METADATA.get(str(thread_id), {})


def get_indexed_threads() -> list:
    """Get list of all threads that have documents indexed."""
    return [tid for tid in _THREAD_RETRIEVERS.keys()]