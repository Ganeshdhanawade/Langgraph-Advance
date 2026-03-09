import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings

from langsmith import traceable #key import

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "RAG_chatbot"
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

model = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()

PDF_PATH = "islr.pdf"

# ------- trace the setup ---------

@traceable(name="load_pdf")
def load_pdf(path:str):
   loader = PyPDFLoader(path)
   return loader.load()  #list[documnets]


@traceable(name="split_document")
def split_documents(docs, chunk_size=1000,chunk_overlap =150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vs = FAISS.from_documents(splits,embeddings)
    return vs

@traceable(name='setup_pipeline')
def setup_pipeline(pdf_path:str,chunk_size=1000,chunk_overlap =150):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs,chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vs = build_vectorstore(splits)
    return vs

# --------------------- pipeline -------------------

prompt = ChatPromptTemplate([
    ('system',"Answwer only form the provided context. If not found, say you don't know."),
    ('human',"quation: {question}\n\ncontext:\n{context}")
])

def format_docs(docs):
    print(docs)
    return "\n\n".join(d.page_content for d in docs)

#build index under traced setup

@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_query(pdf_path: str, question: str):
    # Parent setup run (child of root)
    vectorstore = setup_pipeline(pdf_path, chunk_size=1000, chunk_overlap=150)
    retrivers = vectorstore.as_retriever(search_type="similarity",search_kwargs={'k':4})

    parallel = RunnableParallel({
    "context": retrivers | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
    })

    chain = parallel | prompt | model | parser

    # This LangChain run stays under the same root (since we're inside this traced function)
    lc_config = {"run_name": "pdf_rag_query"}
    return chain.invoke(question, config=lc_config)

if __name__ == "__main__":
    #ask quation
    print("PDF RAG ready. Aks a quation (Ctrl + C to exit).")
    q = input("\nQ: ")
    ans = setup_pipeline_and_query(PDF_PATH, q)
    print("\nA: ", ans)
