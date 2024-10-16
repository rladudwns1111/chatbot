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
openai_api_key = st.secrets["OPENAI_API_KEY"]  # 배포 시 secrets.toml에정 필요 설
if "OPENAI_API_KEY" in st.secrets:
    st.write("API key found")
else:
    st.write("API key missing")

print(openai_api_key)

if __name__ == "__main__":
    st.write("Streamlit App is running")
