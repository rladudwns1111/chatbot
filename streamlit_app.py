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

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None:
    # PyMuPDF를 사용하여 PDF 읽기
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    st.text_area("Extracted Text", text)


st.write(openai_api_key)  # 디버깅 목적으로만 사용


if __name__ == "__main__":
    st.write("Streamlit App is running!")
