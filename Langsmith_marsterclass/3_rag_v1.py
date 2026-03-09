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

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "RAG_chatbot"
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

model = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()

PDF_PATH = "islr.pdf"

#laod_pdf
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

#split
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(docs)

#embedd + index
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vs = FAISS.from_documents(splits,embeddings)
retrivers = vs.as_retriever(search_type="similarity",search_kwargs={'k':4})

#prompt
prompt = ChatPromptTemplate([
    ('system',"Answwer only form the provided context. If not found, say you don't know."),
    ('human',"quation: {question}\n\ncontext:\n{context}")
])

#chain
def format_docs(docs):
    print(docs)
    return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
  "context": retrivers | RunnableLambda(format_docs),
  "question": RunnablePassthrough()
})

chain = parallel | prompt | model | parser

#ask quation
print("PDF RAG ready. Aks a quation (Ctrl + C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip())
print("\nA: ", ans)
