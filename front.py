import requests
import streamlit as st
import json


def handle_user_input(prompt):
    data = {
        "prompt": prompt
    }
    # Convert the dictionary to JSON string
    json_data = json.dumps(data)

    # Set the headers with content type as JSON
    headers = {'Content-Type': 'application/json'}

    # Make the POST request to the local server
    response = requests.post('http://localhost:5000/answer_query', data=json_data, headers=headers)

    try:
        response.raise_for_status()
        print('Request succeeded!')
        print(response.json())
    except requests.exceptions.HTTPError as e:
        print('Request failed!')
        print('Status code:', response.status_code)
        print('Error:', str(e))
    return response.json()




def choose_book(file_name):
    data = {
        "book": file_name
    }
    json_data = json.dumps(data)
    
    headers = {'Content-Type': 'application/json'}
    response = requests.post('http://localhost:5000/choose_doc', data=json_data, headers=headers)
    try:
        response.raise_for_status()
        print('Request succeeded!')
        print(response.json())
    except requests.exceptions.HTTPError as e:
        print('Request failed!')
        print('Status code:', response.status_code)
        print('Error:', str(e))
    return response.json()



def main():
    st.set_page_config(page_title = "Textbook chat", page_icon = ":books:")
    
    st.header("Education Platform")
    

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    user_input = st.chat_input("Ask Anything")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        response = handle_user_input(user_input)
        with st.chat_message("AI"):
            st.write(response["message"])
    
    # with st.sidebar:
    #     book_selection = st.radio("Choose your subject", 
    #                               ["Biology", "National Education"])
    #     file_name = ""
    #     if book_selection == "Biology":
    #         file_name += "biology_12.pdf"
    #     if book_selection == "National Education":
    #         file_name += "national_education_10.pdf"
    #     choose_book(file_name)





if __name__ == '__main__':
    # choose_book("biology_12.pdf")
    main()
