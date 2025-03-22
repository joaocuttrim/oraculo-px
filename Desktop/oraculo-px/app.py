# app.py
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from loaders.load_file import load_file
import os

st.set_page_config(page_title="Oráculo PX - Rocket Lawyer", layout="wide")
st.title("Oráculo PX - Gerenciamento de Projetos")

openai_api_key = st.secrets.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("Chave da API da OpenAI não encontrada. Verifique o secrets.toml.")
    st.stop()

uploaded_file = st.sidebar.file_uploader("Envie um arquivo (PDF, CSV ou TXT)", type=["pdf", "csv", "txt"])

if uploaded_file:
    documents = load_file(uploaded_file)
    if documents is None:
        st.error("Erro ao carregar o arquivo.")
        st.stop()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    chain = load_qa_chain(llm, chain_type="stuff")

    query = st.text_input("Faça sua pergunta sobre o arquivo carregado:")

    if query:
        resposta = chain.run(input_documents=docs, question=query)
        st.write("### Resposta:")
        st.write(resposta)
else:
    st.info("Por favor, envie um arquivo para começar.")
