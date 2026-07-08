import os
import tempfile
import streamlit as st
import pdfplumber

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import requests

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="تحلیل هوشمند گزارش‌های آب و خاک",
    page_icon="💧",
    layout="wide"
)

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;700;800&display=swap');

html, body, [class*="css"]  {
    direction: rtl;
    text-align: right;
    font-family: 'Vazirmatn', sans-serif;
}

.main {
    background: linear-gradient(180deg, #f7fbff 0%, #eef7ff 100%);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.app-header {
    background: linear-gradient(135deg, #0d6efd, #0dcaf0);
    padding: 1.4rem 1.2rem;
    border-radius: 18px;
    color: white;
    box-shadow: 0 8px 24px rgba(13,110,253,0.18);
    margin-bottom: 1rem;
}

.app-header h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 800;
}

.app-header p {
    margin: 0.5rem 0 0;
    font-size: 1rem;
    opacity: 0.95;
}

.info-box {
    background: #ffffff;
    border-right: 5px solid #0d6efd;
    padding: 1rem;
    border-radius: 12px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    margin: 0.8rem 0;
}

.answer-box {
    background: #ffffff;
    border-right: 5px solid #0dcaf0;
    padding: 1.2rem;
    border-radius: 12px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    margin: 0.8rem 0;
    line-height: 1.9;
}

.source-box {
    background: #f8fbff;
    border: 1px dashed #b6dcff;
    padding: 0.8rem;
    border-radius: 10px;
    margin: 0.4rem 0;
    font-size: 0.9rem;
    color: #355070;
}

.footer-box {
    margin-top: 2rem;
    background: #ffffff;
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
    color: #355070;
    box-shadow: 0 4px 14px rgba(0,0,0,0.04);
}

.stButton>button {
    width: 100%;
    border-radius: 12px;
    font-weight: 700;
    padding: 0.7rem 1rem;
}

.stTextInput>div>div>input {
    direction: rtl;
    text-align: right;
}

.stTextArea textarea {
    direction: rtl !important;
    text-align: right !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------
st.markdown("""
<div class="app-header">
    <h1>💧 تحلیل هوشمند گزارش‌های فنی آب و خاک</h1>
    <p>بارگذاری PDF، پردازش سند، و پرسش‌وپاسخ فارسی با هوش مصنوعی</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
این سامانه برای بررسی و تحلیل گزارش‌های فنی، مطالعات آب و خاک، و اسناد تخصصی فارسی طراحی شده است.
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("⚙️ تنظیمات")
    st.caption("برای امنیت، توکن Hugging Face فقط از Streamlit Secrets خوانده می‌شود.")
    st.markdown("""
**نام Secret موردنیاز:**
```toml
HF_TOKEN = "YOUR_HF_TOKEN"
```
توکن رایگان را از این آدرس بساز:
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
(نوع دسترسی: Read کافی است)
""")

    st.divider()

    chunk_size = st.slider("اندازه هر بخش متن (کاراکتر)", 500, 3000, 1000, step=100)
    chunk_overlap = st.slider("همپوشانی بخش‌ها (کاراکتر)", 0, 500, 150, step=50)
    top_k = st.slider("تعداد بخش‌های مرتبط برای پاسخ", 1, 15, 8)
    model_name = st.selectbox(
        "مدل زبانی (متن‌باز، پشتیبان فارسی)",
        [
            "deepseek-ai/DeepSeek-V3-0324",
            "meta-llama/Llama-3.3-70B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
        ],
        index=0,
        help="DeepSeek-V3 معمولاً در تحلیل و ترکیب مطالب از Llama قوی‌تر عمل می‌کند."
    )

    st.divider()
    if st.button("🗑️ پاک‌کردن سند و تاریخچه"):
        for key in ["vectorstore", "chat_history", "processed_filename"]:
            st.session_state.pop(key, None)
        st.rerun()

# ---------------------------
# Helpers
# ---------------------------
def get_hf_token():
    token = st.secrets.get("HF_TOKEN", None)
    if not token:
        st.error("توکن HF_TOKEN در Streamlit Secrets تنظیم نشده است.")
        st.stop()
    return token


@st.cache_resource(show_spinner=False)
def get_embeddings():
    # مدل سبک‌تر مخصوص بازیابی، برای جلوگیری از اتمام حافظه روی سرور رایگان
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small"
    )


def extract_text_from_pdf(file_path: str) -> list[dict]:
    """استخراج متن هر صفحه از PDF به همراه شماره صفحه."""
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i, "text": text})
    return pages


def build_vectorstore(pages: list[dict], chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", "،", " ", ""]
    )

    docs = []
    for p in pages:
        chunks = splitter.split_text(p["text"])
        for chunk in chunks:
            docs.append(Document(
                page_content="passage: " + chunk,
                metadata={"page": p["page"], "raw_text": chunk}
            ))

    if not docs:
        return None

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def answer_question(vectorstore, question: str, hf_token: str, model_name: str, top_k: int):
    relevant_docs = vectorstore.similarity_search("query: " + question, k=top_k)

    context = "\n\n---\n\n".join(
        f"[صفحه {d.metadata.get('page', '?')}]\n{d.metadata.get('raw_text', d.page_content)}"
        for d in relevant_docs
    )

    system_prompt = (
        "شما یک دستیار متخصص و باتجربه در تحلیل گزارش‌های فنی و اسناد حقوقی/قراردادی آب و خاک هستید. "
        "وظیفه‌ی شما ترکیب و تحلیل دقیق اطلاعات از بخش‌های زیر است، نه فقط جستجوی کلمه‌به‌کلمه. "
        "قبل از اینکه بگویید اطلاعاتی موجود نیست، تمام بخش‌های داده‌شده را با دقت بخوان و به دنبال اصطلاحات هم‌معنی، عناوین مشابه، "
        "یا اشاره‌های غیرمستقیم به موضوع سؤال باش (مثلاً اگر سؤال درباره «مهندس ناظر» است، به اصطلاحاتی مانند "
        "«دستگاه نظارت»، «نماینده مقیم»، یا «مهندس مشاور» هم توجه کن، چون معمولاً همان مفهوم را می‌رسانند). "
        "اگر بخش‌هایی از پاسخ در جاهای مختلف متن پراکنده است، آن‌ها را با هم ترکیب کن و یک پاسخ منسجم و ساختاریافته بنویس. "
        "فقط در صورتی که واقعاً هیچ اطلاعات مرتبطی (حتی غیرمستقیم) در متن نیافتی، صادقانه بگو که سند به این موضوع نپرداخته است. "
        "پاسخ را به زبان فارسی روان و دقیق بنویس و در صورت لزوم به شماره صفحه مرتبط اشاره کن.\n\n"
        f"متن سند:\n{context}"
    )

    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "max_tokens": 1000,
        "temperature": 0.2,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        return answer, relevant_docs
    except requests.exceptions.HTTPError as e:
        body = e.response.text if e.response is not None else str(e)
        print(f"[HTTPError] {e} | body={body}")
        st.error(f"خطا در ارتباط با سرویس: {body}")
    except requests.exceptions.RequestException as e:
        print(f"[RequestException] {e}")
        st.error(f"خطا در اتصال شبکه: {e}")
    except (KeyError, IndexError) as e:
        print(f"[ParseError] {e} | raw={resp.text if 'resp' in dir() else 'N/A'}")
        st.error("پاسخ سرویس در قالب مورد انتظار نبود.")
    except Exception as e:
        import traceback
        print(f"[Unexpected Error] {type(e).__name__}: {e}")
        traceback.print_exc()
        st.error(f"خطای غیرمنتظره: {type(e).__name__}: {e}")
    return None, []


# ---------------------------
# Session State Init
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = None

# ---------------------------
# File Upload & Processing
# ---------------------------
st.subheader("📄 بارگذاری گزارش (PDF)")
uploaded_file = st.file_uploader("فایل PDF گزارش را انتخاب کنید", type=["pdf"])

if uploaded_file is not None:
    if st.session_state.processed_filename != uploaded_file.name:
        with st.spinner("در حال پردازش سند... (استخراج متن و ساخت نمایه جستجو)"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                pages = extract_text_from_pdf(tmp_path)
                if not pages:
                    st.warning("متنی در این PDF یافت نشد. ممکن است فایل اسکن‌شده (تصویری) باشد.")
                else:
                    vectorstore = build_vectorstore(pages, chunk_size, chunk_overlap)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.processed_filename = uploaded_file.name
                    st.session_state.chat_history = []
                    st.success(f"سند «{uploaded_file.name}» با موفقیت پردازش شد ({len(pages)} صفحه).")
            finally:
                os.unlink(tmp_path)
    else:
        st.info(f"سند «{uploaded_file.name}» از قبل پردازش شده است.")

# ---------------------------
# Q&A Section
# ---------------------------
if st.session_state.vectorstore is not None:
    st.subheader("💬 پرسش از سند")

    question = st.text_area("سؤال خود را درباره گزارش بنویسید:", height=100, placeholder="مثلاً: میزان شوری خاک در منطقه مورد مطالعه چقدر گزارش شده است؟")

    if st.button("🔍 دریافت پاسخ"):
        if not question.strip():
            st.warning("لطفاً یک سؤال وارد کنید.")
        else:
            hf_token = get_hf_token()
            with st.spinner("در حال تحلیل و تولید پاسخ..."):
                answer, sources = answer_question(
                    st.session_state.vectorstore, question, hf_token, model_name, top_k
                )
            if answer:
                st.session_state.chat_history.insert(0, {
                    "question": question,
                    "answer": answer,
                    "sources": sources
                })

    # نمایش تاریخچه گفتگو
    for item in st.session_state.chat_history:
        st.markdown(f"**❓ {item['question']}**")
        st.markdown(f'<div class="answer-box">{item["answer"]}</div>', unsafe_allow_html=True)
        with st.expander("📚 منابع استفاده‌شده از سند"):
            for d in item["sources"]:
                clean_text = d.metadata.get("raw_text", d.page_content)
                st.markdown(
                    f'<div class="source-box">صفحه {d.metadata.get("page", "?")}: {clean_text[:300]}...</div>',
                    unsafe_allow_html=True
                )
        st.divider()
else:
    st.info("برای شروع، یک فایل PDF بارگذاری کنید.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("""
<div class="footer-box">
ساخته‌شده برای تحلیل تخصصی گزارش‌های آب و خاک 💧 | قدرت‌گرفته از مدل‌های زبانی متن‌باز
</div>
""", unsafe_allow_html=True)
