from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.schema.messages import SystemMessage

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

def main():
    initialize_environment()
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever()

    final_prompt = """\
    You are a helpful, knowledgeable, and smart teaching assistant.
    ...
    Feel free to use any tools available to look up relevant information, only if necessary
    """
    message_final = evaluate_prompt(final_prompt)

if __name__ == "__main__":
    main()
