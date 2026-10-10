import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.llm_setup import get_llm

from pypdf import PdfReader

def process_and_embed_document(uploaded_file):
    """
    Parses, chunks, and embeds an uploaded PDF into a local ChromaDB instance.
    Returns the vector store object.
    """
    # 1. Save uploaded file to a temporary file for PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    try:
        # 2. Parse PDF and enforce 10-page limit
        reader = PdfReader(temp_file_path)
        if len(reader.pages) > 10:
            raise ValueError(f"Document rejected: Found {len(reader.pages)} pages. For optimal performance, please upload a travel guide with 10 pages or fewer.")
            
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        # 3. Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        # Explicitly tag metadata for every chunk
        for split in splits:
            split.metadata["source_file"] = uploaded_file.name
            split.metadata["category"] = "user_upload"

        # 4. Embeddings
        # Using a fast, local Hugging Face model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # 5. ChromaDB Vector Store
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings
        )
        return vectorstore
    finally:
        os.remove(temp_file_path)

def query_rag_document(vectorstore, user_question: str) -> str:
    """
    Retrieves context from ChromaDB and grounds the LLM response in accurate source data.
    """
    llm = get_llm()
    parser = StrOutputParser()
    
    # Retrieve top 3 most relevant chunks with a strict cutoff threshold
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 3}
    )
    docs = retriever.invoke(user_question)
    
    # Combine docs into a single context string
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Grounded Prompt
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Document Assistant.
Answer the user's question USING ONLY the following retrieved context.
If you cannot find the answer in the context, say "I cannot find the answer to that in the provided document." DO NOT hallucinate.

CONTEXT:
{context}"""),
        ("user", "{question}")
    ])
    
    chain = rag_prompt | llm | parser
    
    response = chain.invoke({
        "context": context,
        "question": user_question
    })
    
    return response
