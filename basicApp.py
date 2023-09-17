#add functionality to camera scan pdfs
#use pinecone database
#

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import tnkeeh as tn
from html_template import css, user_template, bot_template
import pyarabic.araby as araby
import re
def _get_all_special_chars():
    ords = (
        list(range(33, 48))
        + list(range(58, 65))
        + list(range(91, 97))
        + list(range(123, 127))
        + [1567, 1548]
    )
    chrs = [chr(num) for num in ords]
    return chrs

def _remove_special_chars(text, excluded_chars=[]):
    
    regex_special_chars = "\\^+*[]-"
    ignored_chars = ""
    for char in excluded_chars:
        if char in regex_special_chars:
            ignored_chars += f"\\" + char
        else:
            ignored_chars += char
    return re.compile("([^\n\u0621-\u064A0-9a-zA-Z " + ignored_chars + "])").sub(
        " ", text
    )

def text_preprocessing(text):
     # im assuming the corpus is taken from twitter, so we need to remove the links and mentions
     text = re.sub(r'http\S+', '', text)
    # remove mentions
     text = re.sub(r'@\S+', '', text)
    # remove hashtags
     text = re.sub(r'#\S+', '', text)
     text = re.sub(r'\d+', '', text)
     #normalize the text(remove harakat, fix hamzat, etc.)
     text = araby.strip_tashkeel(text)
     text = _remove_special_chars(text, _get_all_special_chars())
    #  text = araby.normalize_hamza(text)
    #  text = araby.normalize_ligature(text)
    #  text = araby.normalize_alef(text)
    
     return text


def extract_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

            # cleaned = text_preprocessing(page_text)
            # text += cleaned
    
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator= '\n',
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts = text, embedding = embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
        )
    return conversation_chain

def handle_user_input(user_message):
    response = st.session_state.conversation({"question":user_message})
    st.session_state.chat_history = response["chat_history"]

    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title = "Textbook chat", page_icon = ":books:")
    st.write(css, unsafe_allow_html = True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    
    st.header("Education Platform")

    user_input = st.text_input("Chat Here")
    if user_input:
        handle_user_input(user_input)

    
    with st.sidebar:
        st.subheader("Upload PDF")
        textbooks = st.file_uploader("PDF", accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Loading..."):
                text_content = extract_text(textbooks)

                text_chunks = get_text_chunks(text_content)

                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(vectorstore)






if __name__ == '__main__':
    main()