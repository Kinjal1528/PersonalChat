from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from langchain.docstore.in_memory import InMemoryDocstore
def build_vector_store(pdf_paths: List[str], save_path: str):
    # Load the embedding model
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    all_chunks = []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\nQ", "\nQ", "\n", "."]
        )
        chunks = splitter.split_documents(pages)
        for chunk in chunks:
            chunk.metadata = {"source": os.path.basename(pdf_path)}
        all_chunks.extend(chunks)

    # Convert text chunks to normalized embedding vectors
    texts = [chunk.page_content for chunk in all_chunks]
    raw_vectors = embeddings_model.embed_documents(texts)
    norm_vectors = [np.array(vec) / np.linalg.norm(vec) for vec in raw_vectors]

    # Build FAISS cosine similarity index
    dim = len(norm_vectors[0])
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(norm_vectors).astype('float32'))

    # Create docstore and index map
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(all_chunks)})
    index_to_docstore_id = {i: str(i) for i in range(len(all_chunks))}

    # Create FAISS vectorstore
    vectorstore = FAISS(
        embedding_function=embeddings_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    # Save to disk
    vectorstore.save_local(save_path)
    print(f"✅ Vector store saved to: {save_path}")
def load_vector_store(save_path: str):
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.load_local(
        folder_path=save_path,
        embeddings=embeddings_model,
        allow_dangerous_deserialization=True  # required if using pickle format
    )