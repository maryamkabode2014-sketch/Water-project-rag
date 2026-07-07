
# -*- coding: utf-8 -*-
import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

st.set_page_config(
    page_title="سامانه هوشمند تحلیل فنی پروژه‌های آب و خاک",
    page_icon="💧",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://v1.fontapi.ir/css/Vazir');
    html, body, [class*="css"] {
        font-family: 'Vazir', sans-serif;
        direction: rtl;
        text-align: right;
    }
    .main-title {
        color: #1E3A8A;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        color: #4B5563;
        font-size: 18px;
        text-align: center;
        margin-bottom: 30px;
    }
    .card {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
        border-right: 5px solid #3B82F6;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">💧 سامانه هوشمند تحلیل فنی پروژه‌های آب و خاک</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">تحلیل و انطباق فنی اسناد، آیین‌نامه‌ها و طرح‌های آبیاری بر پایه هوش مصنوعی (Claude)</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### راهنمای مدیران و ناظران فنی")
    st.info("این سامانه برای بارگذاری اسناد فنی و پاسخ‌گویی تحلیلی بر اساس همان اسناد طراحی شده است.")
    st.markdown("---")
    st.markdown("### تنظیمات مدل")
    api_key_input = st.text_input("کلید API Anthropic را وارد کنید:", type="password")
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input

if "db_ready" not in st.session_state:
    st.session_state.db_ready = False
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

st.subheader("۱. بارگذاری اسناد فنی پروژه")
uploaded_files = st.file_uploader(
    "اسناد PDF متنی را بارگذاری کنید (حداکثر ۱۰ فایل)",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    if len(uploaded_files) > 10:
        st.error("لطفاً حداکثر ۱۰ فایل بارگذاری کنید.")
    else:
        st.success(f"{len(uploaded_files)} فایل با موفقیت دریافت شد.")
        if st.button("پردازش اسناد و تحلیل اولیه ⚙️"):
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                st.error("کلید ANTHROPIC_API_KEY یافت نشد. لطفاً آن را در سایدبار وارد کنید.")
            else:
                with st.spinner("در حال پردازش و ساخت پایگاه دانش..."):
                    all_docs = []
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=700,
                        chunk_overlap=120
                    )

                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name

                        loader = PyPDFLoader(tmp_path)
                        pages = loader.load()
                        for p in pages:
                            p.metadata["source_name"] = uploaded_file.name
                        all_docs.extend(pages)

                        os.unlink(tmp_path)

                    chunks = splitter.split_documents(all_docs)

                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )

                    vectordb = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings
                    )

prompt_template = """
تو یک مهندس مشاور و تحلیل‌گر ارشد حوزه آب و خاک و هیدرولیک هستی.
پاسخ‌ها باید کاملاً فنی، مستند و مبتنی بر اسناد بارگذاری‌شده باشند.
اگر پاسخ در اسناد یافت نشد، صراحتاً بگو اطلاعات مربوطه در اسناد موجود نیست.

اسناد:
{context}

سوال کاربر:
{question}

خروجی را در این ساختار ارائه کن:
- خلاصه و نتیجه‌گیری تحلیل
- یافته‌ها و مغایرت‌های فنی
- توصیه‌ها و اقدامات پیشنهادی
- اسناد مرجع
"""
                    prompt = PromptTemplate(
                        template=prompt_template,
                        input_variables=["context", "question"]
                    )

                    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

                    llm = ChatAnthropic(
                        model="claude-3-5-sonnet-latest",
                        temperature=0.15,
                        anthropic_api_key=api_key
                    )

                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": prompt},
                    )

                    st.session_state.db_ready = True
                    st.success("پایگاه دانش ساخته شد. اکنون سوال خود را بپرسید.")

st.markdown("---")
st.subheader("۲. تحلیل فنی و پرسش از اسناد")

user_question = st.text_input(
    "سوال فنی خود را بنویسید:",
    placeholder="مثلاً: آیا ظرفیت پمپ‌ها با نیاز طرح همخوانی دارد؟",
)

if st.button("شروع تحلیل هوشمند 🧠"):
    if not st.session_state.db_ready:
        st.warning("ابتدا اسناد را پردازش کنید.")
    elif not user_question.strip():
        st.warning("لطفاً سوال را وارد کنید.")
    else:
        with st.spinner("در حال تحلیل... لطفاً صبر کنید."):
            response = st.session_state.qa_chain.invoke({"query": user_question})

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📋 نتیجه ارزیابی فنی")
            st.write(response["result"])
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("🔍 مشاهده قطعات استخراج‌شده از اسناد"):
                for i, doc in enumerate(response["source_documents"], 1):
                    file_name = doc.metadata.get("source_name", "سند نامشخص")
                    page_num = doc.metadata.get("page", 0) + 1
                    st.markdown(f"**بخش {i}:** سند: {file_name} | صفحه: {page_num}")
                    st.info(doc.page_content)
