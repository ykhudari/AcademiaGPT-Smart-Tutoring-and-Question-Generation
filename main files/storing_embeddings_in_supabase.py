import vecs
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document

DB_CONNECTION = "postgresql://postgres:supa-jupyteach@192.168.0.77:54328/postgres"

# create vector store client
vx = vecs.create_client(DB_CONNECTION)

# create a collection of vectors with 3 dimensions
docs = vx.get_or_create_collection(name="documents", dimension=1536)


def create_chunks_for_notebook(path):
    loader = TextLoader(path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    
    # add a chunk number and document type
    for i, d in enumerate(docs):
        old = d.metadata
        d.metadata = {**old, "chunk_number": i, "type": "notebook"}
        
    return docs

parsed_notebooks_folder = "./Parsed Notebooks/"

# Listing all markdown files in the folder
markdown_files = [f for f in os.listdir(parsed_notebooks_folder) if f.endswith(".md")]

all_docs = []

# Processing each markdown file in the folder
for markdown_file in markdown_files:
    file_path = os.path.join(parsed_notebooks_folder, markdown_file)
    docs = create_chunks_for_notebook(file_path)
    all_docs.extend(docs)


def create_chunks_for_video(path):
    loader = TextLoader(path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Add a chunk number and document type.
    for i, d in enumerate(docs):
        old = d.metadata
        d.metadata = {**old, "chunk_number": i, "type": "video"}

    return docs


parsed_video_files_folder = "./Embedded Videos/"

# Listing all Python files in the folder
python_files = [f for f in os.listdir(parsed_video_files_folder) if f.endswith(".ipynb")]

all_docs = []

# Processing each Python file in the folder
for python_file in python_files:
    file_path = os.path.join(parsed_video_files_folder, python_file)
    docs = create_chunks_for_video(file_path)
    all_docs.extend(docs)
