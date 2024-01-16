from dotenv import load_dotenv
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.messages import SystemMessage
from langchain.vectorstores.pgvector import PGVector

load_dotenv("/home/jupyteach-msda/jupyteach-ai/.env")

COLLECTION_NAME = "documents"
DB_CONNECTION = "postgresql://postgres:supa-jupyteach@192.168.0.77:54328/postgres"


def get_vectorstore():
    embeddings = OpenAIEmbeddings()

    db = PGVector(embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=DB_CONNECTION,
    )
    return db


def create_chain(system_message_text, temperature=0, model_name="gpt-3.5-turbo-1106"):
    # step 1: create llm
    retriever = get_vectorstore().as_retriever()
    llm = ChatOpenAI(temperature=temperature, model_name=model_name)
    
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
        verbose=False, 
        system_message=system_message
    )


def report_on_message(msg):
    print("any intermediate_steps?: ", len(msg["intermediate_steps"]) > 0)
    print("output:\n", msg["output"])
    print("\n\n")


def chat_and_report(chat, query):
    msg = chat({"input": query})
    report_on_message(msg)
    return msg

def evaluate_prompt(prompt, queries, **kw):
    chat = create_chain(prompt, **kw)
    out = []
    for i, q in enumerate(queries):
        print(f"********** Query {i+1}\n")  
        print(f"input: {q}")
        out.append(chat_and_report(chat, q))
    return out