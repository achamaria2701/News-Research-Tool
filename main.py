import os
import time
import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain



from dotenv import load_dotenv
load_dotenv()

st.title("News Research Tool")
st.sidebar.title("News Article URLs")

folder_path = "faiss_store_openai"
main_placeholder = st.empty()
llm = ChatOpenAI(temperature=0.9, api_key=os.environ['OPENAI_API_KEY'])
embeddings = OpenAIEmbeddings()

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL{i+1}")
    urls.append(url)
process_url_clicked = st.sidebar.button("Process URLs")
if process_url_clicked:
    
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading... Started...")
    data = loader.load()
    main_placeholder.text("Data Splitter... Started...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ['\n\n','\n','.',','],
        chunk_size = 1000
    )
    chunk = text_splitter.split_documents(data)
    main_placeholder.text("Embedding Vector... Started...")
    vectorstore_openai = FAISS.from_documents(chunk, embeddings)
    main_placeholder.text(" Saving vectorstore to local disk...")
    vectorstore_openai.save_local(folder_path)
    time.sleep(2)

query = main_placeholder.text_input("Question: ")
if query:
    main_placeholder.text(" Loading vectorstore from local disk...")
    time.sleep(1)
    vectorstore = FAISS.load_local(folder_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    result = chain({"question": query})
    st.header("Answer")
    st.write(result["answer"])

    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources")
        source_list = sources.split("\n")
        for source in source_list:
            st.write(source)