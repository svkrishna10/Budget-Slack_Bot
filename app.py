import os
import requests
import tqdm
from typing import List

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ACTIVELOOP_TOKEN = os.getenv("DEEPLAKE_TOKEN")



urls = ['https://archive.pib.gov.in/documents/Others/Gbudget2020/ehrel.pdf',
        'https://static.pib.gov.in/WriteReadData/specificdocs/documents/2021/feb/doc20212111.pdf',
        'https://static.pib.gov.in/WriteReadData/userfiles/highlight2022.pdf'
        ]

def load_reports() -> List[str]:
    """ Load pages from a list of urls"""
    pages = []

    for path in os.listdir('D:/Course/LLM/budget_bot/reports'):
        
        loader = PyPDFLoader('D:/Course/LLM/budget_bot/reports/'+str(path))
        local_pages = loader.load_and_split()
        pages.extend(local_pages)
    return pages

pages = load_reports()
print(pages)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(pages)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

#storing in deeplake
db = DeepLake(dataset_path="hub://svkrishna10/unionbudget", embedding_function=embeddings, token=ACTIVELOOP_TOKEN)
db.add_documents(texts)