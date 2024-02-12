import streamlit as st
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
    UnstructuredExcelLoader,
)
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI



def load_document(file_path, file_ext):
    if file_ext == '.pdf':
        return PyPDFLoader(file_path=file_path).load()
    elif file_ext == '.txt':
        return TextLoader(file_path=file_path).load()
    elif file_ext in ['.doc', '.docx']:
        return Docx2txtLoader(file_path=file_path).load()
    elif file_ext == '.ppt':
        return UnstructuredPowerPointLoader(file_path=file_path).load()
    elif file_ext == '.html':
        return UnstructuredHTMLLoader(file_path=file_path).load()
    elif file_ext == '.xls':
        return UnstructuredExcelLoader(file_path=file_path).load()
    elif file_ext == '.csv':
        return CSVLoader(file_path=file_path).load()

def load_multiple_documents(file_infos):
    documents_to_text = []
    documents_to_text.extend(PyPDFLoader(file_path='ASOPs/asop013_133.pdf').load())
    documents_to_text.extend(PyPDFLoader(file_path='ASOPs/asop039_156.pdf').load())
    documents_to_text.extend(PyPDFLoader(file_path='ASOPs/asop043_159.pdf').load())
    for file_path, file_ext in file_infos:
        documents_to_text.extend(load_document(file_path, file_ext))
    return documents_to_text


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

def get_vectorstore(text_chunks, openai_keys):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_keys, model='text-embedding-ada-002')
    vector_store = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
    return vector_store

def create_conservational_chain(vector_store, openai_keys):
    template = """
    You are an AI chatbot with access to a set of PDF documents for context and a deep understanding of the actuarial profession. 
    Users can submit PDFs to help you understand their queries better. 
    Your goal is to provide precise and confident answers by extracting information directly from the submitted PDFs. 
    You should rely on the content within these documents to generate responses. 
    If a user asks a question, your response should be based on the information available in the provided PDFs. 
    Ensure accuracy and confidence in your answers, and if the provided PDFs don't contain relevant information, you can indicate that the query is beyond your current knowledge scope.

    When responding, prioritize accuracy and relevance:
    Context:
    {context}

    Question:
    {question}
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template=template)
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    qa_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=.3, openai_api_key=openai_keys)
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(llm=model, retriever=vector_store.as_retriever(), memory=memory, combine_docs_chain_kwargs={'prompt': qa_prompt})
    return qa_chain


def initialize_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help?"}]
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

def load_qa_chain(saved_files_info, openai_keys):
    loaded_docs = load_multiple_documents(saved_files_info)
    docs_splits = split_documents(loaded_docs)
    vectordb = get_vectorstore(docs_splits, openai_keys)
    return create_conservational_chain(vectordb, openai_keys)