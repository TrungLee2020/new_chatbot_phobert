from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def retriever(pdf_url: str, query: str) -> str:
  docs = PyPDFLoader(pdf_url, extract_images=True).load()
  chroma_db = Chroma.from_documents(docs, embedding=HuggingFaceEmbeddings())

  chroma_retriever = chroma_db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
  retrieverd_docs = chroma_retriever.invoke(query)

  return "\n\n".join([doc.page_content for doc in retrieverd_docs])
