# build_embeddings.py
import os
from retriever import build_vector_store
pdf_folder = r"C:\personalchat\backend\data"

# Gather all PDF file paths
pdf_files = [
    os.path.join(pdf_folder, f)
    for f in os.listdir(pdf_folder)
    if f.lower().endswith(".pdf")
]
 
# Build a single vector DB from all PDFs
build_vector_store(pdf_files, "small_vector_db")
print("✅ Vector DB created from all PDFs.")
