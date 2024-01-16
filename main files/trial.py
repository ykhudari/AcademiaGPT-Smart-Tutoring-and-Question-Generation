from fastapi import FastAPI, HTTPException
from typing import List
import nbformat
import os
import re
import moviepy.editor as mp
import whisper
from whisper.utils import get_writer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import vecs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from helper_functions import embed_documents
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.schema.messages import SystemMessage
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import Query
from langchain.chat_models import ChatOpenAI
from langchain.schema.agent import AgentActionMessageLog


app = FastAPI()

def cells_to_markdown(cells):
    markdown_output = ""
    for cell in cells:
        if cell.cell_type == 'markdown':
            markdown_output += cell.source + '\n\n'
        elif cell.cell_type == 'code':
            markdown_output += f'```python\n{cell.source}\n```\n\n'
    return markdown_output

def convert_nbcontent_to_md(notebook: dict) -> str:
    markdown_content = cells_to_markdown(notebook_content.cells)
    return markdown_content

# Function to convert notebooks to markdown
def convert_notebooks_to_markdown(directory_path):
    if not os.path.exists(directory_path):
        raise HTTPException(status_code=404, detail=f"Directory '{directory_path}' not found.")
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".ipynb"):
            notebook_path = os.path.join(directory_path, filename)
            with open(notebook_path, 'r', encoding='utf-8') as notebook_file:
                notebook_content = nbformat.read(notebook_file, as_version=4)
            
            markdown_content = convert_nbcontent_to_md(notebook_content)
            
            file_name = os.path.splitext(filename)[0]
            md_file_name = f"{file_name}.md"
            
            with open(md_file_name, 'w', encoding='utf-8') as md_file:
                md_file.write(markdown_content)
            
            print(f"Conversion complete for {filename}. Generated {md_file_name}.")
    
    print("Conversion process completed for all notebooks.")

def initialize_environment():
    load_dotenv()

def get_vectorstore():
    COLLECTION_NAME = "documents"
    DB_CONNECTION = "postgresql://postgres:supa-jupyteach@192.168.0.77:54328/postgres"

    embeddings = OpenAIEmbeddings()

    db = PGVector(embedding_function=embeddings,
                  collection_name=COLLECTION_NAME,
                  connection_string=DB_CONNECTION)
    return db
    
'''
# Function to transcribe videos and store in SRT files
def transcribe_videos(transcripts_tiny):
    if not os.path.exists(transcripts_tiny):
        raise HTTPException(status_code=404, detail=f"Directory '{transcripts_tiny}' not found.")
    
    model = whisper.load_model("tiny")
    output_directory = os.path.join(transcripts_tiny, 'transcripts_tiny')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    
    for filename in os.listdir(transcripts_tiny):
        if filename.endswith(".mp4"):
            video_path = os.path.join(transcripts_tiny, filename)
            output_audio_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.mp3")
            
            clip = mp.VideoFileClip(video_path)
            audio_file = clip.audio
            audio_file.write_audiofile(output_audio_path)
            
            result = model.transcribe(output_audio_path)
            
            srt_output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.srt")
            options = {
                'max_line_width': None,
                'max_line_count': None,
                'highlight_words': False
            }
            srt_writer = get_writer("srt", output_directory)
            srt_writer(result, output_audio_path, options)
            
            print(f"Transcription complete for {filename}.")
    
    print("Transcription process completed for all videos.")

# Function to split markdown files
def split_md_files(directory_path):
    if not os.path.exists(directory_path):
        raise HTTPException(status_code=404, detail=f"Directory '{directory_path}' not found.")
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".md"):
            input_file_path = os.path.join(directory_path, filename)
            with open(input_file_path, "r", encoding="utf-8") as file:
                markdown_content = file.read()
            
            ss = RecursiveCharacterTextSplitter(chunk_size=1000)
            split_content = ss.split_text(markdown_content)
            
            output_directory = os.path.join(directory_path, 'split_md_files')
            if not os.path.exists(output_directory):
                os.mkdir(output_directory)
            
            for i, chunk in enumerate(split_content):
                output_file_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_part_{i+1}.md")
                with open(output_file_path, "w", encoding="utf-8") as output_file:
                    output_file.write(chunk)
            
            print(f"Splitting complete for {filename}.")
    
    print("Splitting process completed for all Markdown files.")

# Function to chunk SRT files
def chunk_srt_files(full_text, chunk_length):
    rex = re.compile(r"\d+\n(\d{2}:\d{2}:\d{2}),\d{3} --> (\d{2}:\d{2}:\d{2}),\d{3}")
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
            current_chunk = (
                parts[ix][0],
                part[1],
                current_chunk_text.strip()
            )
            chunks.append(current_chunk)
            ix = i
            current_chunk_text =  part[2]

    return chunks

def process_all_srt_files(input_folder_srt, output_folder_chunk, chunk_length):
    if not os.path.exists(output_folder_chunk):
        os.makedirs(output_folder_chunk)

    srt_files = [f for f in os.listdir(input_folder_srt) if f.endswith('.srt')]

    for srt_file in srt_files:
        with open(os.path.join(input_folder_srt, srt_file)) as f:
            txt = f.read()

        chunks = chunk_srt_files(txt, chunk_length)

        output_file = os.path.join(output_folder_chunk, srt_file)
        with open(output_file, 'w') as f:
            for chunk in chunks:
                f.write(f"{chunk[0]} --> {chunk[1]}\n{chunk[2]}\n\n")

DB_CONNECTION = "postgresql://postgres:supa-jupyteach@192.168.0.77:54328/postgres"
COLLECTION_NAME = "documents"

# create vector store client
vx = vecs.create_client(DB_CONNECTION)

# create a collection of vectors with 3 dimensions
docs = vx.get_or_create_collection(name="documents", dimension=1536)

import re
rex = re.compile(r"\d+\n(\d{2}:\d{2}:\d{2}),\d{3} --> (\d{2}:\d{2}:\d{2}),\d{3}")

def process_notebook(path):
    # step 1: parse notebook to convert .ipynb to .md
    # TODO: need to implement this in a function
    
    # step 2: split markdown into chunks
    docs = create_chunks_for_notebook(path)

    # step 3: create and store embeddings for chunks
    return embed_documents(docs)

def process_videos_in_directory(srt_input_folder):
    # Ensure the directory exists
    if not os.path.exists(srt_input_folder):
        print(f"Directory '{srt_input_folder}' not found.")
        return

    # List all SRT files in the directory
    srt_files = [f for f in os.listdir(srt_input_folder) if f.endswith('.srt')]

    for srt_file in srt_files:
        path_to_transcript = os.path.join(srt_input_folder, srt_file)
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



def create_chain(system_message_text, retriever):
    llm = ChatOpenAI(temperature=0)
    tool = create_retriever_tool(
        retriever,
        "search_course_content",
        "Searches and returns documents regarding the contents of the course and notes from the instructor.",
    )
    tools = [tool]

    system_message = SystemMessage(content=system_message_text)

    return create_conversational_retrieval_agent(
        llm=llm,
        tools=tools,
        verbose=False,
        system_message=system_message
    )

def report_on_message(msg):
    print("any intermediate_steps?: ", len(msg["intermediate_steps"]) > 0)
    print("output:\n", msg["output"])
    print("\n\n")

def chat_and_report(chat_conv, query):
    msg = chat_conv({"input": query})
    report_on_message(msg)
    return msg

def evaluate_prompt(prompt, queries=None, **kw):
    if queries is None:
        queries = []

    chat_conv = create_chain(prompt, **kw)
    out = []
    for i, q in enumerate(queries):
        print(f"********** Query {i+1}\n")
        print(f"input: {q}")
        out.append(chat_and_report(chat_conv, q))
    return out


'''
from typing import List
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.docstore.document import Document
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

class QueryParams(BaseModel):
    query: str
    user_name: str

class ResponseReference(BaseModel):
    chunk_content: str
    metadata: dict


class TutorResponse(BaseModel):
    result: str
    references: List[ResponseReference]

def chunk_and_embed_markdown(directory_path: str):
    load_dotenv()

    DB_CONNECTION = "postgresql://postgres:supa-jupyteach@192.168.0.77:54328/postgres"
    COLLECTION_NAME = "documents"

    # Ensuring the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' not found.")
        return
    
    # Iterating through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".md"):
            input_file_path = os.path.join(directory_path, filename)
            with open(input_file_path, "r", encoding="utf-8") as file:
                markdown_content = file.read()
            
            # Splitting text using RecursiveCharacterTextSplitter
            ss = RecursiveCharacterTextSplitter(chunk_size=1000)
            split_content = ss.split_text(markdown_content)
            
            # Creating a directory to store the split content if it doesn't exist
            output_directory = os.path.join(directory_path, 'split_md_files')
            if not os.path.exists(output_directory):
                os.mkdir(output_directory)
            
            # Writing split content to separate files
            for i, chunk in enumerate(split_content):
                output_file_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_part_{i+1}.md")
                with open(output_file_path, "w", encoding="utf-8") as output_file:
                    output_file.write(chunk)
            
            print(f"Splitting complete for {filename}.")
    
    print("Splitting process completed for all Markdown files.")

    # Embedding the split documents
    documents = []
    for filename in os.listdir(output_directory):
        if filename.endswith(".md"):
            input_file_path = os.path.join(output_directory, filename)
            with open(input_file_path, "r", encoding="utf-8") as file:
                markdown_content = file.read()
                
            metadata = {
                "source": input_file_path,
                # You can add additional metadata here if needed
            }
            doc = Document(page_content=markdown_content, metadata=metadata)
            documents.append(doc)
    
    # Call embedding function
    embedded_db = embed_documents(documents)
    return embedded_db

def embed_documents(docs):
    load_dotenv()
    DB_CONNECTION = "postgresql://postgres:supa-jupyteach@192.168.0.77:54328/postgres"
    COLLECTION_NAME = "documents"

    embeddings = OpenAIEmbeddings()
    db = PGVector.from_documents(
        embedding=embeddings,
        documents=docs,
        collection_name=COLLECTION_NAME,
        connection_string=DB_CONNECTION,
    )
    return db

'''@app.post("/embed_notebook")
async def embed_notebook(x):  # TODO: notebook will be in body of request, need to access
    # do all steps to create 
    # TODO: I do'nt think this method exists... need to find correct one
    notebook = nbformat.parse_string(x["notebook"])
    markdown = convert_nbcontent_to_md(notebook)

    # TODO: make sure we have a function like this defined...
    chunk_and_embed_markdown(markdown)
'''
@app.post("/embed_notebook")
async def embed_notebook(notebook: dict):
    notebook_content = notebook["notebook"]

    # Convert notebook content to markdown
    nb = nbformat.reads(notebook_content, as_version=4)
    markdown = convert_nbcontent_to_md(nb)

    # Chunk and embed markdown content
    chunk_and_embed_markdown(markdown)
    return {"status": 200, "response": "Notebook embedded successfully"}

def create_chain(
        system_message_text, 
        temperature=0, 
        model_name="gpt-3.5-turbo-1106", 
        model_kwargs={"response_format": {"type": "json_object"}},
        verbose=False,
    ):
    # step 1: create llm
    retriever = get_vectorstore().as_retriever()
    llm = ChatOpenAI(temperature=temperature, model_name=model_name, model_kwargs=model_kwargs, verbose=verbose)
    
    # step 2: create retriever tool
    tool = create_retriever_tool(
        retriever,
        "search_course_content",
        "Searches and returns documents regarding the contents of the course and notes from the instructor.",
    )
    tools = [tool]

    # step 3: create system message from the text passed in as an argument
    system_message = SystemMessage(content=system_message_text)

    # return the chain
    return create_conversational_retrieval_agent(
        llm=llm, 
        tools=tools, 
        verbose=verbose, 
        system_message=system_message
    )    
    
'''
"https://api.jupyteach.com/call_ai_tutor?query='some query from the student'&user_name='Spencer Lyon'"
@app.get("/call_ai_tutor")
async def call_ai_tutor(query_params: QueryParams) -> TutorResponse:
    # NOTE: create_chain function from `lyon_common.py` file
    chat = create_chain(FINAL_SYSTEM_PROMPT, ...)
    # TODO: make sure FINAL_SYSTEM_PROMPT has a placeholder for user_name
    output = chat({"input": query_params.query, "user_name": query_params.user_name})
    return TutorResponse(
        result=output["text"],
        # TODO: need to unpack
        references=[
            ResponseReference(
                chunk_content=doc.page_content,
                metadta=doc.metadata
            )
            for doc in output["intermediate_steps"]
        ]
    )
'''
FINAL_SYSTEM_PROMPT = """\
You are a helpful, knowledgeable, and smart teaching assistant.

You are speaking with a student named {user_name}

You specialize in helping students understand concepts their instructors teach by:

1. Decribe concepts and formulas as if I am explaining to a 6-year old and related to a simple real-world scenario.
2. Providing additional examples of the topics being discussed
3. Summarizing content from the instructor, which will be provided to you along with the student's question
4. Reply back with more guidance with concise responses and give a step by step explainations following
the key points that have been covered with more details.
5. Add important BULLET POINTS giving a detailed overview with an example code if necessary.
6. Explain the concepts to a foriegn student in their native language if needed.  
7. Create a sophisticated, humor joke about the concepts and formulas.

Feel free to use any tools available to look up relevant information, only if necessary
"""

app = FastAPI()
@app.get("/call_ai_tutor", response_model=TutorResponse)
async def call_ai_tutor(query: str = Query(None, description="Query text from the student"), user_name: str = Query(None, description="User name")):
    chat = create_chain(FINAL_SYSTEM_PROMPT.format(user_name=user_name), model_kwargs=dict())
    output = chat(query)

    # Extracting text response and intermediate steps for references
    text_response = output.get("output", "")
    intermediate_steps = output.get("intermediate_steps", [])

    references = []
    for step in intermediate_steps:
        if isinstance(step, tuple):
            if len(step) == 2 and isinstance(step[0], AgentActionMessageLog):
                for doc in step[1]:
                    references.append(ResponseReference(
                        chunk_content=doc.page_content,
                        metadata=doc.metadata
                    ))
    return TutorResponse(result=text_response, references=references)







'''
# Define FastAPI endpoint
@app.get("/call_ai_tutor")
async def call_ai_tutor(query_params: QueryParams):
    # Replace these paths with your own
    llm_response = "Demo Response"
    references = [
        {
            "content": "Reference 1 content...",
            "metadata": {"source": "Reference 1 metadata"}
        },
        {
            "content": "Reference 2 content...",
            "metadata": {"source": "Reference 2 metadata"}
        },
    ]
    notebook_dir = 'notebooks/All_notebooks'
    video_transcripts_dir = 'transcripts_tiny'
    markdown_dir = '/path/to/your/directory'
    srt_input_folder = "/videos/transcripts_tiny"
    srt_output_folder = "/videos/transcripts_chunked"
    chunk_length = 1000
    parsed_notebooks_folder = "./Parsed Notebooks/"

    # Listing all markdown files in the folder
    markdown_files = [f for f in os.listdir(parsed_notebooks_folder) if f.endswith(".md")]
    
    all_docs = []
    
    # Processing each markdown file in the folder
    for markdown_file in markdown_files:
        file_path = os.path.join(parsed_notebooks_folder, markdown_file)
        docs = create_chunks_for_notebook(file_path)
        all_docs.extend(docs)

    parsed_video_files_folder = "./Embedded Videos/"

    # Listing all Python files in the folder
    python_files = [f for f in os.listdir(parsed_video_files_folder) if f.endswith(".ipynb")]

    all_docs = []

    # Processing each Python file in the folder
    for python_file in python_files:
        file_path = os.path.join(parsed_video_files_folder, python_file)
        docs = create_chunks_for_video(file_path)
        all_docs.extend(docs)

    convert_notebooks_to_markdown(notebook_dir)
    transcribe_videos(video_transcripts_dir)
    split_md_files(markdown_dir)
    process_all_srt_files(srt_input_folder, srt_output_folder, chunk_length)
    docs = create_chunks_for_notebook(path)
    embed_documents(docs)

    initialize_environment()
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever()

    final_prompt = """\
    You are a helpful, knowledgeable, and smart teaching assistant.

    You specialize in helping students understand concepts their instructors teach by:

    1. Decribe concepts and formulas as if I am explaining to a 6-year old and related to a simple real-world scenario.
    2. Providing additional examples of the topics being discussed
    3. Summarizing content from the instructor, which will be provided to you along with the student's question
    4. Reply back with more guidance with concise responses and give a step by step explainations following
    the key points that have been covered with more details.
    5. Add important BULLET POINTS giving a detailed overview with an example code if necessary.
    6. Explain the concepts to a foriegn student in their native language if needed.  
    7. Create a sophisticated, humor joke about the concepts and formulas.
    
    Feel free to use any tools available to look up relevant information, only if necessary
    """
    message_final = evaluate_prompt(final_prompt)
    
    return {"status": 200, "response": "All data processing complete."}
'''