import streamlit as st
from openai import OpenAIError
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser

# OpenAI API 키 자동으로 설정
openai_api_key = st.secrets["OPENAI_API_KEY"]
st.write(openai_api_key)
############################### 1단계 : OpenAI를 활용한 질문 처리 ##########################

# RAG 체인 생성 함수
def get_rag_chain() -> Runnable:
    template = """
    다음의 컨텍스트를 활용해서 질문에 답변해줘
    - 질문에 대한 응답을 해줘
    - 간결하게 5줄 이내로 해줘
    - 곧바로 응답결과를 말해줘

    질문: {question}

    응답:"""
    
    custom_rag_prompt = PromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    
    return custom_rag_prompt | model | StrOutputParser()

# 사용자 질문 처리
def process_question(user_question):
    try:
        chain = get_rag_chain()
        response = chain.invoke({"question": user_question})
        return response
    except OpenAIError as e:
        st.error(f"Error with OpenAI: {str(e)}")
        return None

############################### Streamlit 앱 메인 함수 ##########################
def main():
    st.title("자동 OpenAI API 연결 Q&A 서비스")

    # 사용자 질문 입력
    user_question = st.text_input("질문을 입력하세요", placeholder="무엇이든 질문해보세요!")

    if user_question:
        st.subheader("질문에 대한 답변")
        response = process_question(user_question)
        if response:
            st.text(response)

if __name__ == "__main__":
    main()
