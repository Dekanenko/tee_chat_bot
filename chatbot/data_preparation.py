from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from os.path import exists

def get_vecdb(embedding_function, data_path="chatbot/data/docs/", persist_directory="chatbot/data/vecdb/"):
    if exists(persist_directory):
        vectordb = Chroma(
            embedding_function=embedding_function,
            persist_directory=persist_directory)
        
        return vectordb
    
# parse the FAQ.txt
    loader = TextLoader(data_path+"/FAQ.txt")
    faq = loader.load()

    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=0, 
        separators=["\n\n"]
    )
    faq_splits = r_splitter.split_documents(faq)

# parse the tshirts_customization.txt
    loader = TextLoader(data_path+"/tshirts_customization.txt")
    tcustom = loader.load()

    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0, 
    )
    tcustom_splits = r_splitter.split_documents(tcustom)

    faq_splits.append(tcustom_splits[0])
    vectordb = Chroma.from_documents(
        documents=faq_splits,
        embedding=embedding_function,
        persist_directory=persist_directory
    )

    return vectordb