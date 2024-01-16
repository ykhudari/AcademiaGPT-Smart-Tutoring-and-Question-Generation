import os
import textwrap
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.schema.messages import SystemMessage
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.schema.agent import AgentActionMessageLog
from fastapi import FastAPI, Query, Header, HTTPException, Depends
from typing import List
from enum import Enum
from langchain.output_parsers import PydanticOutputParser
from pydantic_settings import BaseSettings
import pickle
import pathlib
from structlog import get_logger


class Settings(BaseSettings):
    db_connection: str = (
        "postgresql://postgres:supa-jupyteach@192.168.0.77:54328/postgres"
    )
    openai_api_key: str
    api_token: str


settings = Settings()
pickle_dir = pathlib.Path(os.path.dirname(__file__)) / "pickles"
pickle_dir.mkdir(exist_ok=True, parents=True)
logger = get_logger()
app = FastAPI()


def save_chat_memory(chat, chat_id):
    with open(pickle_dir / f"{chat_id}.pickle", "wb") as f:
        mem = chat.memory.chat_memory
        pickle.dump(mem, f)
        logger.msg(f"saved {len(mem.messages)} messages", chat_id=chat_id)


def maybe_load_chat_memory(chat, chat_id):
    file_name = pickle_dir / f"{chat_id}.pickle"
    if file_name.exists():
        with open(file_name, "rb") as f:
            mem = pickle.load(f)
            chat.memory.chat_memory = mem
            logger.msg(f"loaded {len(mem.messages)} messages", chat_id=chat_id)
    return chat


# Dependency to validate the token
async def token_dependency(x_token: str = Header(...)):
    if x_token != settings.api_token:
        raise HTTPException(status_code=403, detail="Invalid API Token")
    return x_token


def get_vectorstore(course_slug):
    embeddings = OpenAIEmbeddings()

    db = PGVector(
        embedding_function=embeddings,
        collection_name=course_slug,
        connection_string=settings.db_connection,
    )
    return db


class QueryParams(BaseModel):
    query: str
    user_name: str


class ResponseReference(BaseModel):
    chunk_content: str
    metadata: dict


class TutorResponse(BaseModel):
    result: str
    references: List[ResponseReference]


class GeneratorResponse(BaseModel):
    result: str
    references: List[ResponseReference]


class MessageRole(str, Enum):
    user = "user"
    system = "system"
    assistant = "assistant"


class Message(BaseModel):
    content: str
    role: MessageRole


def create_chain(
    system_message_text,
    course_slug,
    temperature=0,
    model_name="gpt-4-1106-preview",
    model_kwargs={"response_format": {"type": "json_object"}},
    verbose=False,
):
    # step 1: create llm
    retriever = get_vectorstore(course_slug=course_slug).as_retriever()
    llm = ChatOpenAI(
        temperature=temperature,
        model_name=model_name,
        model_kwargs=model_kwargs,
        verbose=verbose,
    )

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
        llm=llm, tools=tools, verbose=verbose, system_message=system_message
    )


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


class TutorRequest(BaseModel):
    query: str
    course_slug: str
    chat_id: int
    user_name: str | None = None


@app.post(
    "/call_ai_tutor",
    response_model=TutorResponse,
    dependencies=[Depends(token_dependency)],
)
async def call_ai_tutor(
    request: TutorRequest,
) -> TutorResponse:
    # unpack request
    query = request.query
    course_slug = request.course_slug
    chat_id = request.chat_id
    user_name = request.user_name

    log = logger.bind(
        course_slug=course_slug,
        chat_id=chat_id,
        user_name=user_name,
        method="call_ai_tutor",
    )
    chat = create_chain(
        FINAL_SYSTEM_PROMPT.format(user_name=user_name),
        model_kwargs=dict(),
        course_slug=course_slug,
    )

    log.msg("Chain created")
    chat = maybe_load_chat_memory(chat, chat_id)
    output = chat(query)
    log.msg("ran query", query=query, output_keys=", ".join(list(output.keys())))
    save_chat_memory(chat, chat_id)

    # Extracting text response and intermediate steps for references
    text_response = output.get("output", "")
    intermediate_steps = output.get("intermediate_steps", [])

    references = []
    for step in intermediate_steps:
        if (
            isinstance(step, tuple)
            and len(step) == 2
            and isinstance(step[0], AgentActionMessageLog)
        ):
            for doc in step[1]:
                references.append(
                    ResponseReference(
                        chunk_content=doc.page_content, metadata=doc.metadata
                    )
                )
    return TutorResponse(result=text_response, references=references)


# Fucntion that returns the system prompt with the format of the question requested
def create_system_prompt(pydantic_object):
    common_system_prompt = textwrap.dedent(
        """
    You are a smart, helpful teaching assistant chatbot named AcademiaGPT.

    You are an expert Python programmer and have used all the most popular
    libraries for data analysis, machine learning, and artificial intelligence.

    You assist professors that teach courses about Python, data science, and machine learning
    to college students.

    Your task is to help professors produce practice questions to help students solidify
    their understanding of specific topics

    In your conversations with a professor, you  will be given a topic and an
    expected difficulty level (integer) or (string).

    If the difficulty is not given assume the difficulty level to be the previously used difficulty level.

    Here is an example question with difficulty 1

    {{
        "question_text":"How would you reverse the order of the following list in python\n\n```python\na = [1, 'hi', 3, 'there']\n```\n\nand save the result in an object `b`",
        "starting_code":"a = [1, 'hi', 3, 'there']\n# Reverse the order of the list and save the result in an object called b",
        "solution":"a = [1, 'hi', 3, 'there']\nb = a[::-1]",
        "topics":["python","programming","lists"],
        "difficulty":1,
        "setup_code":"# none",
        "test_code":"assert b == ['there', 3, 'hi', 1]"
    }}


    Here is an example question with difficulty 2

    {{
        "question_text": "Given a list of stock prices `prices` for consecutive days, write a for loop that calculates the total return of the stock over the period. The total return is defined as the percentage change from the first day to the last day. Store the result in a variable named `total_return`.",
        "starting_code": "prices = [100, 102, 105, 110, 108]\n# Calculate the total return and store it in total_return",
        "solution": "prices = [100, 102, 105, 110, 108]\nfirst_price = prices[0]\nlast_price = prices[-1]\ntotal_return = ((last_price - first_price) / first_price) * 100",
        "topics": ["for loops", "asset pricing"],
        "difficulty": 2,
        "setup_code": "# No setup code required",
        "test_code": "assert abs(total_return - ((prices[-1] - prices[0]) / prices[0]) * 100) < 1e-6"
    }}



    Here is an example question with difficulty 3

    {{
        "question_text":"You are given a 3 dimensional numpy array as specified below:\n\n```\nA = np.array([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]])\n```\n\nCreate a variable `idx` (define as a tuple) that you could use to select the `4.0` element of this array.\n\nFor example,\n\n```\nidx = (0, 0, 0)\n```\n\nwould select the `0.0` element of the array.",
        "starting_code":"idx = (0, 0, 0)  # Fill this in with the correct index",
        "solution":"x = (1, 0, 0)",
        "topics":["numpy"],
        "difficulty":3,
        "setup_code":"import numpy as np\n\nA = np.array([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]])",
        "test_code":"assert A[idx] == A[1, 0, 0]"
    }}

    If you are asked to give similar, easier, or another question, the user wants you to use the same topic and difficulty
    level that you used to generate the previous question.

    You are encouraged to use any tools available to look up relevant information, only
    if necessary.

    Your responses must always exactly match the specified JSON format with no extra words or content.

    YOU MUST ALWAYS PRODUCE EXACTLY ONE JSON OBJECT.

    {format_instructions}
    """
    )

    parser = PydanticOutputParser(pydantic_object=pydantic_object)
    return common_system_prompt.format(
        format_instructions=parser.get_format_instructions()
    )


@app.get(
    "/call_question_generator",
    response_model=GeneratorResponse,
    dependencies=[Depends(token_dependency)],
)
async def call_question_generator(
    query: str,
    course_slug: str = "documents",
    user_name: str = Query(None, description="User name"),
) -> GeneratorResponse:
    system_prompt = create_system_prompt(QueryParams(query=query, user_name=user_name))
    chat = create_chain(
        system_prompt.format(user_name=user_name),
        model_kwargs=dict(),
        course_slug=course_slug,
    )
    output = chat(query)

    # Extracting text response and intermediate steps for references
    text_response = output.get("output", "")
    intermediate_steps = output.get("intermediate_steps", [])

    references = []
    for step in intermediate_steps:
        if (
            isinstance(step, tuple)
            and len(step) == 2
            and isinstance(step[0], AgentActionMessageLog)
        ):
            for doc in step[1]:
                references.append(
                    ResponseReference(
                        chunk_content=doc.page_content, metadata=doc.metadata
                    )
                )
    return GeneratorResponse(result=text_response, references=references)
