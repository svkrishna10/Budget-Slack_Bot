import os
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ACTIVELOOP_TOKEN = os.getenv("DEEPLAKE_TOKEN")

# Initialize Embeddings and Dataset
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
financial_report_dataset = 'hub://svkrishna10/unionbudget'

#Retrieval Q&A
db = DeepLake(dataset_path=financial_report_dataset, embedding=embeddings, token=ACTIVELOOP_TOKEN,read_only=True) 
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0,openai_api_key= OPENAI_API_KEY), chain_type="stuff", retriever=db.as_retriever(), return_source_documents=True)

app = App(token=os.getenv("SLACK_BOT_TOKEN"))  

@app.message("")
def message(message, say):
    response = qa.invoke(message["text"])
    
    if 'result' in response:
        result_summary = response['result']
        say(text=result_summary)
    else:
        say(text="Unable to retrieve information. Please try again.")


#Start your app
if __name__ == "__main__":
    SocketModeHandler(app,os.getenv("SLACK_APP_TOKEN")).start() 