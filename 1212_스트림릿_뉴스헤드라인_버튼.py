import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# Streamlit 캐시 설정
@st.cache_resource
def load_embeddings():
    return OpenAIEmbeddings()

@st.cache_resource
def load_vector_db():
    embeddings = load_embeddings()
    return FAISS.load_local("faiss_dataframe_db_fixed", embeddings=embeddings, allow_dangerous_deserialization=True)

@st.cache_resource
def load_llm():
    return ChatOpenAI(model_name="gpt-4o", temperature=0.5)

def generate_unique_headlines(_headline_chain, news_content):
    try:
        # 헤드라인 생성
        response = _headline_chain.run(news_content=news_content)
        
        # 생성된 헤드라인들을 리스트로 분리
        headlines = [h.strip() for h in response.split('\n') if h.strip()]
        
        # 중복 제거 및 4개로 제한
        unique_headlines = list(dict.fromkeys(headlines))[:4]
        
        return unique_headlines
    except Exception as e:
        st.error(f"헤드라인 생성 중 오류 발생: {e}")
        logging.error(f"헤드라인 생성 오류: {e}")
        return []

def get_weekly_headlines(start_date, end_date, _vector_db):
    week_start = start_date
    week_end = week_start + timedelta(days=6)

    logging.info(f"검색 날짜 범위: {week_start} - {week_end}")

    try:
        # 모든 문서 검색
        results = _vector_db.similarity_search_with_score("", k=1000)
        
        # 날짜 필터링
        filtered_results = [
            (doc, score) for doc, score in results 
            if doc.metadata.get('date') and 
               week_start.strftime('%Y-%m-%d') <= doc.metadata['date'] <= week_end.strftime('%Y-%m-%d')
        ]

        logging.info(f"필터링된 결과 수: {len(filtered_results)}")
        
        if filtered_results:
            # 필터링된 문서들의 내용 결합
            news_content = "\n\n".join([doc.page_content for doc, _ in filtered_results])
            
            return news_content
        else:
            st.warning("해당 주차에 대한 뉴스 콘텐츠가 없습니다.")
            return None
    
    except Exception as e:
        st.error(f"헤드라인 검색 중 오류 발생: {e}")
        logging.error(f"헤드라인 검색 오류: {e}")
        return None

def get_month_week_number(date):
    # 해당 월의 주차를 구하는 함수
    first_day_of_month = date.replace(day=1)
    first_weekday_of_month = first_day_of_month.weekday()
    
    if first_weekday_of_month == 6:
        first_week_start = first_day_of_month
    else:
        first_week_start = first_day_of_month - timedelta(days=first_weekday_of_month + 1)

    delta_days = (date - first_week_start).days
    week_number = delta_days // 7 + 1

    return f"{date.year}년 {date.month:02d}월 {week_number}주차"

def main():
    # 리소스 로드 (캐시 활용)
    embeddings = load_embeddings()
    vector_db = load_vector_db()
    llm = load_llm()

    # 헤드라인 생성을 위한 프롬프트 템플릿
    headline_prompt = PromptTemplate(
        input_variables=["news_content"],
        template="""다음 뉴스 기사들의 핵심 내용을 바탕으로 간결하고 매력적인 헤드라인 4개를 생성해주세요. 
        각 헤드라인은 고유하고 서로 다른 주요 포인트를 강조해야 합니다.

    뉴스 기사 내용:
    {news_content}

    지침:
    - 간결하고 명확한 헤드라인 작성
    - 중복 없이 고유한 내용 강조
    - 한국어로 작성
    - 각 헤드라인은 다른 관점이나 측면을 다뤄야 함
    """
    )

    # 헤드라인 생성 체인
    headline_chain = LLMChain(llm=llm, prompt=headline_prompt)

    st.title("Weekly News Headlines")

    # 세션 상태 초기화
    if 'current_date' not in st.session_state:
        st.session_state['current_date'] = datetime(2024, 1, 1)

    # 버튼을 통한 이동
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("이전 주"):
            st.session_state['current_date'] -= timedelta(weeks=1)
            st.rerun()

    with col3:
        if st.button("다음 주"):
            st.session_state['current_date'] += timedelta(weeks=1)
            st.rerun()

    # 현재 선택된 날짜로 헤드라인 표시
    current_date = st.session_state['current_date']
    
    # 월별 주차 정보 표시
    week_title = get_month_week_number(current_date)
    st.subheader(week_title)

    # 헤드라인 가져오기
    news_content = get_weekly_headlines(current_date, current_date + timedelta(days=6), vector_db)
    
    if news_content:
        # 고유한 헤드라인 생성
        unique_headlines = generate_unique_headlines(headline_chain, news_content)

        # 헤드라인 출력
        if unique_headlines:
            for headline in unique_headlines:
                st.write(f"{headline}")
                st.write("---")
        else:
            st.warning("생성된 헤드라인이 없습니다.")
    else:
        st.warning("해당 주차에 대한 뉴스 헤드라인을 찾을 수 없습니다.")

    # 디버깅을 위한 추가 정보
    st.write(f"현재 선택된 날짜: {current_date}")
    st.write(f"필터링 날짜 범위: {current_date} - {current_date + timedelta(days=6)}")

if __name__ == '__main__':
    main()