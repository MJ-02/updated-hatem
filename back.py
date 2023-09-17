import requests
from flask import Flask, request, jsonify
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.embeddings import CohereEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chains.question_answering import load_qa_chain
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.agents import Tool
from langchain.agents import initialize_agent

OPENAI_API_KEY = "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"
PINECONE_API_KEY = "200041bf-c55e-4fa5-bcc8-3e986f7b2766"
PINECONE_API_ENV = "northamerica-northeast1-gcp"
COHERE_API_KEY = "pyBY3uGmd1kZgkjgQDkIGoZQUZMPVkoCToLkVLRb"



app = Flask(__name__)

#initialize the embedding model
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# def process_pdf(path):
#     loader = PyPDFLoader(path)
#     data = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
#     texts = text_splitter.split_documents(data)
#     embeddings = CohereEmbeddings(model="embed-multilingual-v2.0", cohere_api_key= COHERE_API_KEY)
#     pinecone.init(
#         api_key=PINECONE_API_KEY, 
#         environment=PINECONE_API_ENV  
#     )
#     index_name = "crownprincecomp"
#     docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
#     return docsearch
pinecone.init(
        api_key=PINECONE_API_KEY, 
        environment=PINECONE_API_ENV  
    )
def process_pdf(path):
    #load the pdf into the enviroment
    loader = PyPDFLoader(path)
    data = loader.load()
    #spilt the pdf text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(data)
    
    #init the vector db and index
    pinecone.init(
        api_key=PINECONE_API_KEY, 
        environment=PINECONE_API_ENV  
    )
    index_name = "smallchunks"
    #vectorize the data
    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
    return docsearch
    


# docs = process_pdf("C:/Users/majal/Desktop/Grace Datathon/example pdfs/NDgwOTA5MQ8686_2019_pdf.pdf")
# docs = process_pdf("example pdfs/Ch.01_Introduction_ to_computers.pdf")


def llm_chain(query, docs):
    docs = docs.similarity_search(query)
    llm = OpenAI(temperature=0.2, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=query)
    return response



@app.route('/process_data', methods=['POST'])
def process_data():
    data = request.get_json()
    user_query = data["prompt"]
    # response = llm_chain(user_query, docs)
    response = chat_agent(user_query)
    response_data = {'message': response}
    return jsonify(response_data), 200

# def learn_more():
#     llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm = llm,
#         retriever = vectorstore.as_retriever(),
#         memory = memory
#         )
#     return conversation_chain


@app.route('/learn_more', methods= ['POST'])
def chat_agent(prompt):
    text_field = 'text'
    index = pinecone.Index("smallchunks")
    vectorstore = Pinecone(index, embeddings.embed_query, text_field)

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                    model="gpt-3.5-turbo",
                    temperature=0
                    )

    conversational_memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k = 5,
        return_messages=True
    )

    qa = RetrievalQA.from_chain_type(
        llm= llm, 
        chain_type="stuff",
        retriever = vectorstore.as_retriever()
    )
    tools = [
    Tool(
        name='Knowledge Base',
        func=qa.run,
        description=(
            "Answer user questions using the knowledge base and produce an output in the arabic"
        )
    )
    ]
    agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
    )
    return agent(f"أنت مساعد ذكاء اصطناعي وسوف تتلقى سؤالًا من المستخدم بلغات مختلفة، يجب أن تجيب فقط باللغة العربية ولا شيء غير ذلك. السؤال هو: {prompt}")["output"]



    # tool = create_retriever_tool(
    #     retriever, 
    #     "search_textbooks",
    #     "Searches and returns answers from textbooks."
    # )
    # tools = [tool]
    # agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)
    # result = agent_executor({"input":"ما هي الاحزاب"})
    # return result["output"]






if __name__ == '__main__':
    app.run()