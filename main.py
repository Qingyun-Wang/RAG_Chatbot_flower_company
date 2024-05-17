from chat_chain import *

# import streamlit as st

from dotenv import load_dotenv
load_dotenv()


def main():
    st.title("Chatbot for Flower Company")
    image_path = 'static/flower.png'
    st.image(image_path)

    # Initialize chat history in session_state if it doesn't already exist
    if 'chat_history_record' not in st.session_state:
        st.session_state.chat_history_record = ChatMessageHistory()

    # Use session_state for storing and accessing the chat history
    chat_history_record = st.session_state.chat_history_record

    chat_chain = build_chat_chain('OneFlower')

    question = st.text_input("Enter your question")

    if question:
        answer = chat_chain.invoke({"input": question}, {"configurable": {"session_id": "qy"}})['answer']
        # st.write('chat history:', chat_history_record.messages)
        st.write("Answer:", answer)

# Run the Streamlit app
if __name__ == "__main__":
    main()


