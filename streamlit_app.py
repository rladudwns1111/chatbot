from http.client import responses

## streamlit 관련 모듈 불러오기
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
import os
import fitz  # PyMuPDF
import re

## 환경변수 불러오기
# Streamlit secrets 사용하여 OpenAI API 키 불러오기
openai_api_key = st.secrets["OPENAI_API_KEY"]  # 배포 시 secrets.toml에 설정 필요


############################### 1단계 : PDF 문서를 벡터DB에 저장하는 함수들 ##########################
## 내가 파일을 업로드 하면 PDF_임시폴더 라는 폴더에 파일들이 저장되게 됨
## 내가 파일을 parsing 하던지 chunking 하던지 할 때 파일이 우선 컴퓨터에 저정되어있어야 가능하기 때문
## 1: 임시폴더에 파일 저장
def save_uploadedfile(uploadedfile: UploadedFile) -> str:
    temp_dir = "PDF_임시폴더"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.read())
    return file_path


## List안에 Document들이 여러개 있는 형태로 변환을 해주는 함수
## 경로를 불러와서 경로를 기반으로 PyMuPDFLoader 라는 랭체인에서 제공해주는 모듈을 사용해서
## PDF 파일을 읽고 이 정보들을 Document에 추가하고 결론적으로 List[Document]를 만들어줌
## 2: 저장된 PDF 파일을 Document로 변환
def pdf_to_documents(pdf_path: str) -> List[Document]:
    documents = []
    loader = PyMuPDFLoader(pdf_path)
    doc = loader.load()
    for d in doc:
        d.metadata['file_path'] = pdf_path
    documents.extend(doc)
    return documents


## TextSplitter라는 것을 이용해소 Document를 더 작게 쪼갬
## 그 후 다시 그것을 Document로 반환함
## 3: Document를 더 작은 document로 변환
def chunk_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    ## 원래는 PDF파일 1개를 기준으로 저장이 되는데 글자수 800글자를 기준으로하고 누락되는 정보가 없도록 앞뒤로는 100자 정도 겹치게 해서 저장함
    return text_splitter.split_documents(documents)


## faiss를 사용해서
## 4: Document를 벡터DB로 저장
def save_to_vector_store(documents: List[Document]) -> None:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    ## 어떤 AI 모델을 사용하여 임베딩을 할것인지 선언해준 것
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")
    ## faiss_index 라는 곳에 저장함


############################### 2단계 : RAG 기능 구현과 관련된 함수들 ##########################


## 사용자 질문에 대한 RAG 처리
@st.cache_data
def process_question(user_question):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

    ## 벡터 DB 호출
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    ## 관련 문서 3개를 호출하는 Retriever 생성
    retriever = new_db.as_retriever(search_kwargs={"k": 3})
    ## 사용자 질문을 기반으로 관련문서 3개 검색
    retrieve_docs: List[Document] = retriever.invoke(user_question)

    ## RAG 체인 선언
    chain = get_rag_chain()
    ## 질문과 문맥을 넣어서 체인 결과 호출
    response = chain.invoke({"question": user_question, "context": retrieve_docs})

    return response, retrieve_docs


def get_rag_chain() -> Runnable:
    template = """
    다음의 컨텍스트를 활용해서 질문에 답변해줘
    - 질문에 대한 응답을 해줘
    - 간결하게 5줄 이내로 해줘
    - 곧바로 응답결과를 말해줘

    컨텍스트 : {context}

    질문: {question}

    응답:"""

    custom_rag_prompt = PromptTemplate.from_template(template)
    model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

    return custom_rag_prompt | model | StrOutputParser()


############################### 3단계 : 응답결과와 문서를 함께 보도록 도와주는 함수 ##########################
@st.cache_data(show_spinner=False)
def convert_pdf_to_images(pdf_path: str, dpi: int = 250) -> List[str]:
    doc = fitz.open(pdf_path)  # 문서 열기
    image_paths = []

    # 이미지 저장용 폴더 생성
    output_folder = "PDF_이미지"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for page_num in range(len(doc)):  # 각 페이지를 순회
        page = doc.load_page(page_num)  # 페이지 로드

        zoom = dpi / 72  # 72이 디폴트 DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)  # type: ignore

        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")  # 페이지 이미지 저장 page_1.png, page_2.png, etc.
        pix.save(image_path)  # PNG 형태로 저장
        image_paths.append(image_path)  # 경로를 저장

    return image_paths


def display_pdf_page(image_path: str, page_number: int) -> None:
    image_bytes = open(image_path, "rb").read()  # 파일에서 이미지 인식
    st.image(image_bytes, caption=f"Page {page_number}", output_format="PNG", width=600)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]


def main():
    # st.text(dotenv_values(".env"))
    # st.text("셋팅완료")

    pdf_doc = st.file_uploader("PDF Uploader", type="pdf")
    button = st.button("PDF 업로드하기")
    if pdf_doc and button:
        with st.spinner("PDF 문서 저장중"):
            st.text("여기까지구현됨")
            pdf_path = save_uploadedfile(
                pdf_doc)  # save_uploadedfile 이라는 함수를 사용해서 업로드한 PDF 파일을 내부적으로 pdf_path 라는 경로에 저장해줌
            pdf_document = pdf_to_documents(pdf_path)  # 내가 저장한 경로를 기반으로 document를 만들어줌
            smaller_documents = chunk_documents(pdf_document)  # 페이지 하나당 1개의 document가 나오기 때문에 더 작은 document로 쪼개줌
            save_to_vector_store(smaller_documents)  # 이 쪼개진 document 들을 vector db에 저장해줌

    user_question = st.text_input("PDF 문서에 대해서 질문해 주세요",
                                  placeholder="무순위 청약 시에도 부부 중복신청이 가능한가요?")

    if user_question:
        response, context = process_question(user_question)
        st.text(response)
        st.text(context)


if __name__ == "__main__":
    main()
