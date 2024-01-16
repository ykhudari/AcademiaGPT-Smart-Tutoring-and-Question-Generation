import openai
import os
import dotenv
from typing import List
from dotenv import load_dotenv
from langchain.schema.document import Document
from helper_functions import embed_documents
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings.openai import OpenAIEmbeddings

import vecs

DB_CONNECTION = "postgresql://postgres:supa-jupyteach@192.168.0.77:54328/postgres"
COLLECTION_NAME = "documents"

# create vector store client
vx = vecs.create_client(DB_CONNECTION)

# create a collection of vectors with 3 dimensions
docs = vx.get_or_create_collection(name="documents", dimension=1536)

import re
rex = re.compile(r"\d+\n(\d{2}:\d{2}:\d{2}),\d{3} --> (\d{2}:\d{2}:\d{2}),\d{3}")

def chunk_srt_files(full_text, chunk_length):

    splits = rex.split(full_text)[1:]

    parts = []
    for i in range(0, len(splits), 3):
        start_time = splits[i]
        end_time = splits[i+1]
        content = splits[i+2].strip()
        parts.append((start_time, end_time, content))
        
    chunks = []
    ix = 0
    current_chunk_text = ""
    for i, part in enumerate(parts):
        current_chunk_text = current_chunk_text + " " + part[2]
        if len(current_chunk_text) > chunk_length or i == len(parts) - 1:
            # if we have a long enough chunk OR we are on the last piece of content...
            current_chunk = (
                parts[ix][0],  # starting timestamp
                part[1],
                current_chunk_text.strip()
            )
            chunks.append(current_chunk)
            ix = i  # we repeat this chunk one more time for overlap
            current_chunk_text =  part[2]

    return chunks

def read_txt_file(file_path):
    """Read the content of a .txt file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# NOTE: use the srt files directly from whisper, not ones we have modified
# you can find these in this folder: jupyteach-ai/videos/transcripts_tiny
'''
def process_video(path_to_transcript):
    # step 1: read in the srt_content from the file at `path_to_transcript`
    srt_content = read_txt_file(path_to_transcript)

    # step 2: call `chunk_srt_files` (see above) on the srt_content
    chunks = chunk_srt_files(srt_content, 1000)
    
    # step 3: convert the (start, end, txt) tuples you get back from `chunk_srt_files`
    #         into langchain.schema.document.Document with metadata set properly
    docs = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "source": path_to_transcript, 
            "chunk_number": i, 
            "timestamps": f"{chunk[0]} --> {chunk[1]}"
        }
        doc = Document(page_content = chunk[2], metadata=metadata)
        docs.append(doc)
        
    # step 4: call `embed_documents` to create/store emebddings for the video
    return embed_documents(docs)

pandas_intro = process_video("videos/transcripts_tiny/2.1.1 pandas intro.srt")
'''

def process_videos_in_directory(directory_path):
    # Ensure the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' not found.")
        return

    # List all SRT files in the directory
    srt_files = [f for f in os.listdir(directory_path) if f.endswith('.srt')]

    for srt_file in srt_files:
        path_to_transcript = os.path.join(directory_path, srt_file)
        srt_content = read_txt_file(path_to_transcript)
        chunks = chunk_srt_files(srt_content, 1000)
        
        docs = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": path_to_transcript, 
                "chunk_number": i, 
                "timestamps": f"{chunk[0]} --> {chunk[1]}"
            }
            doc = Document(page_content=chunk[2], metadata=metadata)
            docs.append(doc)
            
        embed_documents(docs)

# Replace 'directory_path' with the path of the directory containing your .srt files
directory_path = 'videos/transcripts_tiny/'
process_videos_in_directory(directory_path)