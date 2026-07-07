import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA

# ----------------------------
# صفحه و استایل
# ----------------------------
st.set_page_config(page_title="سامانه تحلیل گزارش‌های فنی آب", layout="wide")

st.markdown("""
<style>
    .main {
        direction: rtl;
        text-align: right;
        font-family: Vazirmatn, Arial, sans-serif;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    .card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

st.title("💧 سامانه تحلیل گزارش‌های فنی آب")

st.write(
    "فایل PDF گزارش را بارگذاری کنید، سپس روی «پردازش اسناد» کلیک کنید و سوال خود را بپرسید."
)

# ----------------------------
# کلید API
# ----------------------------
def get_openai_key():
    return (
        os.environ.get("OPENAI_API_KEY")
        or st.secrets.get("OPENAI_API_KEY", "")
        or ""
    )

openai_api_key = get_openai_key()

with st.sidebar:
    st.header("تنظیمات")

    api_key_input = st.text_input(
        "کلید OpenAI API را وارد کنید:",
        type="password",
        value=openai_api_key if openai_api_key else "",
    )

    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
        openai_api_key = api_key_input

    st.markdown("---")
    st.caption("اگر کلید را در Streamlit Secrets گذاشته‌اید، اینجا لازم نیست دوباره وارد کنید.")

# ----------------------------
# اعتبارسنجی کلید
# ----------------------------
if not openai_api_key:
    st.warning("کلید OPENAI_API_KEY یافت نشد. لطفاً آن را در سایدبار یا Secrets وارد کنید.")
    st.stop()

# ----------------------------
# آپلود فایل
# ----------------------------
uploaded_files = st.file_uploader(
    "فایل‌های PDF را بارگذاری کنید",
    type=["pdf"],
    accept_multiple_files=True
)

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "source_docs" not in st.session_state:
    st.session_state.source_docs = []

# ----------------------------
# پردازش اسناد
# ----------------------------
if st.button("پردازش اسناد"):
    if not uploaded_files:
        st.error("لطفاً حداقل یک فایل PDF بارگذاری کنید.")
        st.stop()

    try:
        all_docs = []

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name

            loader = PyPDFLoader(temp_path)
            docs = loader.load()

            for d in docs:
                d.metadata["source_name"] = uploaded_file.name

            all_docs.extend(docs)
            os.unlink(temp_path)

        if not all_docs:
            st.error("هیچ متنی از PDF استخراج نشد. احتمالاً فایل اسکن‌شده یا تصویری است.")
            st.stop()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=120,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(all_docs)

        if not chunks:
            st.error("هیچ chunkی ساخته نشد. محتوای PDF قابل پردازش نیست.")
            st.stop()

        st.info(f"{len(all_docs)} صفحه/سند خوانده شد و {len(chunks)} بخش ساخته شد.")

        embeddings = OpenAIEmbeddings(
            api_key=openai_api_key,
            model="text-embedding-3-small"
        )

        vectorstore = FAISS.from_documents(chunks, embeddings)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4o-mini",
            temperature=0.15
        )

        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff"
        )

        st.session_state.source_docs = chunks

        st.success("اسناد با موفقیت پردازش شدند. حالا می‌توانید سوال بپرسید.")

    except Exception as e:
        st.error(f"خطا در پردازش اسناد: {e}")
        st.stop()

# ----------------------------
# پرسش از اسناد
# ----------------------------
if st.session_state.qa_chain:
    user_question = st.text_input("سوال خود را درباره اسناد وارد کنید:")

    if st.button("پرسش"):
        if not user_question.strip():
            st.warning("لطفاً یک سوال وارد کنید.")
            st.stop()

        try:
            response = st.session_state.qa_chain.invoke({"query": user_question})

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### پاسخ")
            st.write(response.get("result", "پاسخی دریافت نشد."))
            st.markdown("</div>", unsafe_allow_html=True)

            source_documents = response.get("source_documents", [])
            if source_documents:
                with st.expander("مشاهده منابع"):
                    for i, doc in enumerate(source_documents, 1):
                        file_name = doc.metadata.get("source_name", "سند نامشخص")
                        page_num = doc.metadata.get("page", 0) + 1
                        st.markdown(f"**منبع {i}:** `{file_name}` | صفحه {page_num}")
                        st.info(doc.page_content)

        except Exception as e:
            st.error(f"خطا در زمان پاسخ‌دهی: {e}")
else:
    st.info("ابتدا اسناد را پردازش کنید.")
