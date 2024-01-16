from typing import List

from dotenv import load_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.docstore.document import Document

load_dotenv()

DB_CONNECTION = "postgresql://postgres:supa-jupyteach@192.168.0.77:54328/postgres"
COLLECTION_NAME = "documents"


def embed_documents(docs: List[Document]):
    embeddings = OpenAIEmbeddings()

    # this will create and store the actual embeddings for the docs
    db = PGVector.from_documents(
        embedding=embeddings,
        documents=docs,
        collection_name=COLLECTION_NAME,
        connection_string=DB_CONNECTION,
    )
    return db


def get_vectorstore():
    embeddings = OpenAIEmbeddings()

    db = PGVector(embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=DB_CONNECTION,
    )
    return db

# # For Retreiving Question Generator 

# import requests
# from pydantic import BaseModel, Field
# import datetime

# # Define the input schema
# class AcademicsAI(BaseModel):
#     tutor: float = Field(..., description="provide help to college students")
#     questions: float = Field(..., description="generate random answer that are asked by students about data science")

# @tool(args_schema=AcademicsAI)
# def get_current_temperature(tutor: float, questions: float) -> dict:
#     """Expain more about Data Prepartion."""
    
#     #BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
#     # Parameters for the request
#     params = {
#         'tutor': tutor,
#         'questions': questions,
#         'hourly': 'questions_asked_1m',
#         'forecast_days': 2,
#     }
#     # Make the request
#     response = requests.get(BASE_URL, params=params)
    
#     if response.status_code == 200:
#         results = response.json()
#     else:
#         raise Exception(f"API Request failed with status code: {response.status_code}")

#     current_utc_time = datetime.datetime.utcnow()
#     time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in results['hourly']['time']]
#     temperature_list = results['hourly']['temperature_2m']
    
#     closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
#     current_temperature = temperature_list[closest_time_index]
    
#     return f'The current temperature is {current_temperature}°C'