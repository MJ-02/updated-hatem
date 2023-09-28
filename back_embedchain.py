# import requests
from flask import Flask, request, jsonify
# from langchain.document_loaders import PyPDFLoader
# from langchain.vectorstores import Pinecone
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.llms import OpenAI
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferWindowMemory
# from langchain.chains import ConversationalRetrievalChain, RetrievalQA
# from langchain.chains.question_answering import load_qa_chain
# import pinecone
# from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
# from langchain.agents import Tool
# from langchain.agents import initialize_agent
from embedchain import App
from embedchain.config import ChromaDbConfig
from datetime import datetime
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

db_config = ChromaDbConfig(chroma_settings={"allow_reset":True})

app = Flask(__name__)

#initialize app
tawjihi_bot = App()
now  = datetime.now()

document_dict = {
    "biology": ["biology_12.pdf", "biology_12_summary.pdf"],
    "history": ["jordan_history_12.pdf", "jordan_history_12_1_summary.pdf"]
}

def log(msg:str) -> None:
    with open("logs.txt", "a") as f:
        f.write(f"{msg} --- {now.strftime('%d/%m/%Y %H:%M:%S')}\n")

log("New Session Started")


#Recieves Json from POST request in the form of {"subject": subject}
#returns Book chosen with code 200
@app.route('/choose_doc', methods = ["POST"])
def choose_doc() -> jsonify:
    try:
        data = request.get_json()
        sub = data["subject"]
        docs = document_dict[sub]
        for book in docs:
            tawjihi_bot.add(f"example_pdfs/{book}", "pdf_file")
    except Exception as e:
        return jsonify("Internal Server Error"), 500
    else:
        return jsonify("Book chosen!"), 200
    


#Recieves Json from POST request in the form of {"prompt": user_message}
#Return Json in the form of {"message":response} with code 200
@app.route('/answer_query', methods=['POST'])
def answer_query() -> jsonify:
    try:
        data = request.get_json()
        user_query = data["prompt"]
        response = tawjihi_bot.chat(user_query)
        response_data = {'message': response}
    except Exception as e:
        return jsonify("Internal Server Error"), 500
    else:
        return jsonify(response_data), 200


if __name__ == '__main__':
    app.run()
