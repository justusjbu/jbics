import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from PIL import Image
import time
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

import os
from dotenv import load_dotenv

# Load .env file (only needed locally; hosted platforms inject vars automatically)
load_dotenv()

# Access variables
#api_key_old = os.getenv("OPENAI_API_KEY")
api_key1 = st.secrets["OPENAI_API_KEY"]


def type_message(message, placeholder):
    """Display message with typing animation."""
    displayed_message = ""
    for i in range(0, len(message), 2):
        next_chars = message[i : i + 2]
        displayed_message += next_chars
        placeholder.markdown(displayed_message + "▌")
        time.sleep(0.01)  # Adjust speed here (lower = faster)
    placeholder.markdown(displayed_message)

image = Image.open('JBU-Letter-Logo.png')

st.image(image)


st.title("JBICS JBU Chatbot")


#From PDF
loader = PyPDFLoader("jbu-2025.pdf")
#loader = PyPDFLoader("BIble-kjv.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)



#print(texts)

embeddings = OpenAIEmbeddings(api_key=api_key1)
db = Chroma.from_documents(texts, embeddings)

if "messages" not in st.session_state:
    model = "gpt-4o-mini",
    st.session_state.messages = [
        {
            "role": "developer",
            "content": "You are a kind and good assistant. You are also a bit rude and will not hesitate to tell the user to behave properly if they are being annoying.",
        }
    ]

    # Display chat messages
for message in st.session_state.messages[1:]:  # Skip the system prompt
    with st.chat_message(message["role"]):
        st.write(message["content"])

    # User input
if user_input := st.chat_input("Ask a Question about JBU?"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)


#user_question = st.text_input("Please type your question here:", key="question")

llm1= ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=api_key1
)

if user_input and user_input !="":

    qa = RetrievalQA.from_chain_type(llm=llm1, chain_type="stuff", retriever=db.as_retriever())

    query = user_input
    result = qa.invoke(query)
    #   st.write(result)

    # Get and display assistant response with typing animation
    with st.chat_message("assistant"):
        if result:
            placeholder = st.empty()
            type_message(result, placeholder)
            st.session_state.messages.append(
                {"role": "assistant", "content": result}
            )
























