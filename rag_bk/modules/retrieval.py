from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader


def retriever():
    # PDF 파일 로드. 파일의 경로 입력

    loader = PyPDFLoader("data/paper.pdf")

    # 텍스트 분할기를 사용하여 문서를 분할합니다.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # 문서를 로드하고 분할합니다.
    split_docs = loader.load_and_split(text_splitter)

    # VectorStore를 생성합니다.
    vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())

    # Retriever를 생성합니다.
    retriever = vector.as_retriever()

    return retriever
