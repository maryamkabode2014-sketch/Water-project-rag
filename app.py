import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA

# تنظیمات صفحه
st.set_page_config(page_title="سامانه تحلیل گزارش‌های فنی آب", layout="wide")

st.markdown("""
<style>
    .main { direction: rtl; text-align: right; font-family: Arial; }
    .stButton>button { background-color: #d97706; color: white; width: 100%; }
</style>
""", unsafe_allow_html=True)

st.title("💧 سامانه تحلیل گزارش‌های فنی آب (Claude 3.5)")

# دریافت کلید Anthropic از Secrets یا سایدبار
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")

with st.sidebar:
    st.header("تنظیمات")
    api_key_input = st.text_input("کلید Anthropic API:", type="password", value=anthropic_api_key)
    if api_key_input:
        anthropic_api_key = api_key_input

if not anthropic_api_key:
    st.warning("کلید ANTHROPIC_API_KEY یافت نشد. لطفاً آن را در سایدبار یا Secrets وارد کنید.")
    st.stop()

uploaded_files = st.file_uploader("فایل PDF گزارش را بارگذاری کنید", type=["pdf"], accept_multiple_files=True)

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if st.button("پردازش اسناد"):
    if not uploaded_files:
        st.error("فایلی انتخاب نشده است.")
    else:
        try:
            all_docs = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    loader = PyPDFLoader(tmp.name)
                    docs = loader.load()
                    for d in docs: d.metadata["source"] = uploaded_file.name
                    all_docs.extend(docs)
                os.unlink(tmp.name)

            # تقسیم متن
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.split_documents(all_docs)

            # استفاده از مدل Embedding رایگان (نیاز به کلید ندارد)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # ذخیره در حافظه موقت FAISS
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            # تنظیم مدل Claude
            llm = ChatAnthropic(
                anthropic_api_key=anthropic_api_key,
                model_name="claude-3-5-sonnet-20240620",
                temperature=0.1
            )

            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
                return_source_documents=True
            )
            st.success("اسناد با موفقیت پردازش شدند. آماده پاسخگویی!")

        except Exception as e:
            st.error(f"خطا در پردازش: {e}")

# بخش پرسش و پاسخ
if st.session_state.qa_chain:
    query = st.text_input("سوال فنی خود را بپرسید:")
    if st.button("تحلیل") and query:
        res = st.session_state.qa_chain.invoke({"query": query})
        st.markdown("### 📋 پاسخ نهایی:")
        st.write(res["result"])
        
        with st.expander("منابع استخراج شده"):
            for doc in res["source_documents"]:
                st.write(f"📄 منبع: {doc.metadata['source']} | محتوا: {doc.page_content[:200]}...")
