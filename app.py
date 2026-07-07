import os
import streamlit as st
import pdfplumber

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA

# ----------------------------
# تنظیمات اولیه صفحه
# ----------------------------
st.set_page_config(
    page_title="سامانه هوشمند تحلیل گزارش‌های فنی آب و خاک",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# استایل سفارشی
# ----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;700;800&display=swap');

    html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        direction: rtl;
        text-align: right;
        font-family: 'Vazirmatn', sans-serif;
    }

    .main > div {
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }

    .hero-box {
        background: linear-gradient(135deg, #0f766e 0%, #0369a1 50%, #1d4ed8 100%);
        padding: 2rem 2rem;
        border-radius: 22px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.95;
        line-height: 1.9;
    }

    .info-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.04);
    }

    .dev-card {
        background: linear-gradient(135deg, #ecfeff 0%, #f0fdf4 100%);
        border: 1px solid #bae6fd;
        border-radius: 18px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    .dev-name {
        font-size: 1.1rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.3rem;
    }

    .dev-role {
        font-size: 0.9rem;
        color: #0f766e;
    }

    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 12px;
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: white;
        font-weight: 700;
        padding: 0.75rem 1rem;
        transition: 0.3s ease;
        box-shadow: 0 6px 14px rgba(3,105,161,0.25);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
    }

    div[data-testid="stFileUploader"] {
        background: #f8fafc;
        border: 1px dashed #94a3b8;
        border-radius: 16px;
        padding: 1rem;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }

    .footer-box {
        margin-top: 2rem;
        padding: 1rem;
        text-align: center;
        color: #475569;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# هدر
# ----------------------------
st.markdown("""
<div class="hero-box">
    <div class="hero-title">💧 سامانه هوشمند تحلیل گزارش‌های فنی آب و خاک</div>
    <div class="hero-subtitle">
        تحلیل، خلاصه‌سازی و پاسخ‌گویی هوشمند به گزارش‌های تخصصی با استفاده از مدل Claude
        و بازیابی محتوای اسناد برای استخراج دقیق اطلاعات از فایل‌های PDF
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# دریافت API KEY
# ----------------------------
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")

with st.sidebar:
    st.markdown("""
    <div class="dev-card">
        <div style="font-weight:700; color:#0369a1;">توسعه‌دهنده سامانه</div>
        <div class="dev-name">مریم کبوده</div>
        <div class="dev-role">حوزه آب، خاک و تحلیل هوشمند اسناد فنی</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ تنظیمات")
    api_key_input = st.text_input(
        "کلید Anthropic API",
        type="password",
        value=anthropic_api_key,
        help="در صورت تمایل می‌توانید کلید را از Secrets یا این بخش وارد کنید."
    )

    if api_key_input:
        anthropic_api_key = api_key_input

    st.markdown("""
    <div class="info-card">
        <b>راهنما</b><br>
        1) فایل PDF را بارگذاری کنید.<br>
        2) روی «پردازش اسناد» بزنید.<br>
        3) سپس سوال فنی خود را مطرح کنید.
    </div>
    """, unsafe_allow_html=True)

if not anthropic_api_key:
    st.warning("⚠️ کلید ANTHROPIC_API_KEY یافت نشد. لطفاً آن را در Secrets یا سایدبار وارد کنید.")
    st.stop()

# ----------------------------
# آپلود فایل
# ----------------------------
st.markdown('<div class="section-title">📂 بارگذاری اسناد</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "فایل‌های PDF گزارش‌های فنی را انتخاب کنید",
    type=["pdf"],
    accept_multiple_files=True
)

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "docs_count" not in st.session_state:
    st.session_state.docs_count = 0

if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0

# ----------------------------
# تابع خواندن PDF
# ----------------------------
def extract_text_from_pdfs(files):
    all_docs = []

    for uploaded_file in files:
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                for i, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            all_docs.append(
                                Document(
                                    page_content=text,
                                    metadata={
                                        "source": uploaded_file.name,
                                        "page": i + 1
                                    }
                                )
                            )
                    except Exception:
                        # اگر یک صفحه مشکل داشت، برنامه متوقف نشود
                        continue
        except Exception:
            # اگر یک فایل مشکل داشت، بقیه فایل‌ها ادامه پیدا کنند
            continue

    return all_docs

# ----------------------------
# پردازش اسناد
# ----------------------------
if st.button("🚀 پردازش اسناد"):
    if not uploaded_files:
        st.error("❌ لطفاً حداقل یک فایل PDF بارگذاری کنید.")
    else:
        with st.spinner("در حال خواندن فایل‌ها و آماده‌سازی پایگاه دانش..."):
            try:
                all_docs = extract_text_from_pdfs(uploaded_files)

                if not all_docs:
                    st.error("⚠️ هیچ متن قابل خواندنی در فایل‌ها یافت نشد. ممکن است PDFها اسکن تصویری باشند یا متن قابل استخراج نداشته باشند.")
                    st.stop()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=900,
                    chunk_overlap=120
                )
                chunks = splitter.split_documents(all_docs)

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                vectorstore = FAISS.from_documents(chunks, embeddings)

                llm = ChatAnthropic(
                    anthropic_api_key=anthropic_api_key,
                    model_name="claude-3-5-sonnet-latest",
                    temperature=0.1
                )

                st.session_state.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
                    return_source_documents=True
                )

                st.session_state.docs_count = len(all_docs)
                st.session_state.chunks_count = len(chunks)

                st.success(
                    f"✅ پردازش با موفقیت انجام شد. "
                    f"{len(all_docs)} صفحه متنی و {len(chunks)} بخش دانشی آماده تحلیل هستند."
                )

            except Exception as e:
                st.error(f"❌ خطا در پردازش اسناد: {e}")

# ----------------------------
# نمایش وضعیت
# ----------------------------
if st.session_state.qa_chain is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📄 تعداد صفحات پردازش‌شده: {st.session_state.docs_count}")
    with col2:
        st.info(f"🧩 تعداد قطعه‌های متنی: {st.session_state.chunks_count}")

# ----------------------------
# پرسش و پاسخ
# ----------------------------
if st.session_state.qa_chain:
    st.markdown("---")
    st.markdown('<div class="section-title">💬 پرسش از اسناد</div>', unsafe_allow_html=True)

    query = st.text_input(
        "سوال تخصصی خود را وارد کنید",
        placeholder="مثلاً: جمع‌بندی وضعیت منابع آب زیرزمینی در این گزارش چیست؟"
    )

    if st.button("🔍 تحلیل و پاسخ"):
        if not query.strip():
            st.warning("لطفاً سوال خود را وارد کنید.")
        else:
            with st.spinner("در حال تحلیل اسناد و تولید پاسخ..."):
                try:
                    result = st.session_state.qa_chain.invoke({"query": query})

                    st.markdown("### 📋 پاسخ نهایی")
                    st.write(result["result"])

                    with st.expander("📚 منابع و صفحات استفاده‌شده"):
                        for doc in result["source_documents"]:
                            source_name = doc.metadata.get("source", "نامشخص")
                            page_num = doc.metadata.get("page", "نامشخص")
                            preview = doc.page_content[:300].replace("\n", " ")

                            st.markdown(f"**فایل:** `{source_name}` | **صفحه:** `{page_num}`")
                            st.write(preview + "...")
                            st.markdown("---")

                except Exception as api_err:
                    st.error(f"❌ خطا در ارتباط با سرویس Claude: {api_err}")
                    st.info("اگر خطا ادامه داشت، اعتبار کلید API و وضعیت حساب Anthropic را بررسی کنید.")

# ----------------------------
# فوتر
# ----------------------------
st.markdown("""
<div class="footer-box">
    توسعه داده شده توسط <b>مریم کبوده</b> |
    سامانه تخصصی تحلیل گزارش‌های فنی آب، خاک و اسناد مهندسی 💧
</div>
""", unsafe_allow_html=True)
