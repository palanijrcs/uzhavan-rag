import io
import os
import base64
import hashlib
import streamlit as st
from gtts import gTTS
from openai import OpenAI
from rag_engine import AgriculturalRAG, BilingualSchemeResponse

st.set_page_config(
    page_title="Uzhavan RAG | உழவன் வழிகாட்டி",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling: Keeps selectboxes clean and uncluttered
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #2E7D32; margin-bottom: 0px; }
    .sub-header { font-size: 1.05rem; color: #555; margin-bottom: 20px; }
    .tamil-text { font-family: 'Mukta Malar', sans-serif; font-size: 1.05rem; line-height: 1.6; }
    div[data-testid="stSidebar"] div[data-testid="stSelectbox"] input {
        caret-color: transparent !important;
        cursor: pointer !important;
    }
    </style>
""", unsafe_allow_html=True)

DEFAULT_SCHEMES = {
    "All": {
        "levels": ["All", "Central", "Tamil Nadu"],
        "categories": ["All", "income_support", "credit", "insurance"]
    },
    "PM-KISAN": {
        "levels": ["Central"],
        "categories": ["income_support"]
    },
    "KCC": {
        "levels": ["Central"],
        "categories": ["credit"]
    },
    "Crop Insurance (PMFBY)": {
        "levels": ["Central", "Tamil Nadu"],
        "categories": ["insurance"]
    },
    "Kalaignar Scheme (TN)": {
        "levels": ["Tamil Nadu"],
        "categories": ["income_support"]
    }
}


@st.cache_resource(show_spinner="Initializing RAG Knowledge Bases...")
def get_rag_engine():
    return AgriculturalRAG()


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_configured" not in st.session_state:
        st.session_state.api_configured = bool(os.getenv("OPENAI_API_KEY"))
    if "last_processed_audio_hash" not in st.session_state:
        st.session_state.last_processed_audio_hash = None
    if "simulated_voice_prompt" not in st.session_state:
        st.session_state.simulated_voice_prompt = None
    if "current_pdf_id" not in st.session_state:
        st.session_state.current_pdf_id = None
    if "auto_scheme_name" not in st.session_state:
        st.session_state.auto_scheme_name = ""
    if "auto_jurisdiction" not in st.session_state:
        st.session_state.auto_jurisdiction = "Central"
    if "auto_category" not in st.session_state:
        st.session_state.auto_category = "income_support"
    if "scheme_catalog" not in st.session_state:
        st.session_state.scheme_catalog = DEFAULT_SCHEMES.copy()


def generate_tamil_audio(text: str) -> io.BytesIO:
    fp = io.BytesIO()
    tts = gTTS(text=text, lang="ta", slow=False)
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp


def transcribe_audio(audio_file) -> str:
    client = OpenAI()
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    return transcription.text


def render_sidebar(engine: AgriculturalRAG):
    st.sidebar.image("https://img.icons8.com/color/96/tractor.png", width=64)
    st.sidebar.markdown("### **Uzhavan Scheme AI**\n*உழவன் விவசாய தகவல் மையம்*")
    st.sidebar.divider()

    if not st.session_state.api_configured:
        key = st.sidebar.text_input("OpenAI API Key", type="password")
        if key:
            os.environ["OPENAI_API_KEY"] = key
            st.session_state.api_configured = True
            st.sidebar.success("Key validated!")
            st.rerun()
        else:
            st.sidebar.warning("Provide an API Key to query.")
            st.stop()

    # Upload Official Scheme PDF with Strict Auto-Extraction Display
    st.sidebar.subheader("📄 Upload Official Scheme PDF")
    uploaded_pdf = st.sidebar.file_uploader("Drop Central/TN Gazette PDF", type=["pdf"])

    if uploaded_pdf is not None:
        file_bytes = uploaded_pdf.getvalue()
        pdf_file_id = f"{uploaded_pdf.name}_{len(file_bytes)}"

        # Run extraction only when a newly dropped file is detected
        if st.session_state.current_pdf_id != pdf_file_id:
            with st.spinner("Inspecting gazette headers with LLM..."):
                auto_meta = engine.extract_pdf_metadata(file_bytes, uploaded_pdf.name)
                st.session_state.auto_scheme_name = auto_meta.scheme_name
                st.session_state.auto_jurisdiction = auto_meta.jurisdiction if auto_meta.jurisdiction in ["Central", "Tamil Nadu"] else "Central"
                st.session_state.auto_category = auto_meta.category if auto_meta.category in ["income_support", "credit", "insurance"] else "income_support"
                st.session_state.current_pdf_id = pdf_file_id

        with st.sidebar.expander("👁️ Verify Uploaded PDF Content", expanded=False):
            file_size_kb = round(len(file_bytes) / 1024, 1)
            st.markdown(f"**File:** `{uploaded_pdf.name}` ({file_size_kb} KB)")
            base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="220" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

        target_scheme = st.sidebar.text_input(
            "Detected Scheme Name",
            value=st.session_state.auto_scheme_name or uploaded_pdf.name.replace(".pdf", "")
        )

        manual_override = st.sidebar.checkbox("Manual Metadata Override", value=False)

        if not manual_override:
            # Strictly display ONLY the extracted values with no other dropdown clutter
            pdf_level = st.session_state.auto_jurisdiction
            pdf_cat = st.session_state.auto_category

            st.sidebar.selectbox(
                "Detected Jurisdiction",
                options=[pdf_level],
                index=0,
                disabled=True,
                help="Automatically deduced from official gazette text."
            )
            st.sidebar.selectbox(
                "Detected Category",
                options=[pdf_cat],
                index=0,
                disabled=True,
                help="Automatically deduced from official gazette text."
            )
        else:
            # Dropdowns activated only when user checks manual override
            pdf_level = st.sidebar.selectbox(
                "Jurisdiction Override",
                ["Tamil Nadu", "Central"],
                index=0 if st.session_state.auto_jurisdiction == "Tamil Nadu" else 1
            )
            pdf_cat = st.sidebar.selectbox(
                "Category Override",
                ["income_support", "credit", "insurance"],
                index=["income_support", "credit", "insurance"].index(st.session_state.auto_category)
            )

        if st.sidebar.button("Index Uploaded PDF"):
            with st.spinner("Extracting chunks & vectorizing..."):
                chunk_count = engine.ingest_pdf_file(
                    file_bytes,
                    uploaded_pdf.name,
                    target_scheme,
                    pdf_level,
                    pdf_cat
                )
                st.session_state.scheme_catalog[target_scheme] = {
                    "levels": [pdf_level],
                    "categories": [pdf_cat]
                }
                st.sidebar.success(f"Indexed {chunk_count} chunks from '{target_scheme}'!")
                st.rerun()

    st.sidebar.divider()

    # Guided Filters
    st.sidebar.subheader("🎯 Guided Filters")
    catalog = st.session_state.scheme_catalog
    scheme_filter = st.sidebar.selectbox("Target Scheme", list(catalog.keys()), index=0)
    
    allowed_levels = catalog[scheme_filter]["levels"]
    allowed_categories = catalog[scheme_filter]["categories"]
    
    level_filter = st.sidebar.selectbox("Jurisdiction", allowed_levels, index=0)
    category_filter = st.sidebar.selectbox("Support Category", allowed_categories, index=0)

    st.sidebar.divider()
    st.sidebar.subheader("📥 Baseline Schemes")
    
    if st.sidebar.button("Load Baseline Knowledge (All 4 Schemes)"):
        st.session_state.simulated_voice_prompt = None
        with st.spinner("Indexing baseline agricultural schemes..."):
            baseline_docs = [
                {
                    "text": (
                        "Pradhan Mantri Kisan Samman Nidhi (PM-KISAN) is a Central Sector Scheme providing income support "
                        "to all landholding farmers' families in the country. Under the Scheme, Rs 6000 per year is released "
                        "directly into the bank accounts of beneficiaries in three equal installments of Rs 2000 every 4 months. "
                        "Eligible beneficiaries are small and marginal landholder farmer families with cultivable land. "
                        "Exclusions apply to institutional landholders, government employees, and income-tax payees."
                    ),
                    "metadata": {
                        "source": "pmkisan.gov.in",
                        "scheme_name": "PM-KISAN",
                        "level": "Central",
                        "category": "income_support"
                    }
                },
                {
                    "text": (
                        "Kisan Credit Card (KCC) scheme provides farmers with timely access to short-term credit for cultivating "
                        "crops, purchasing seeds and fertilizers, and meeting post-harvest expenses. The credit limit is fixed "
                        "based on landholding, cropping pattern, and scale of finance. Loans up to Rs 3 Lakh receive an interest "
                        "subvention of 2% and prompt repayment incentive of 3%, bringing the effective interest rate to 4% per annum. "
                        "All farmers, individual/joint borrowers, tenant farmers, sharecroppers, and Self Help Groups are eligible."
                    ),
                    "metadata": {
                        "source": "agricoop.nic.in (KCC Portal)",
                        "scheme_name": "KCC",
                        "level": "Central",
                        "category": "credit"
                    }
                },
                {
                    "text": (
                        "Pradhan Mantri Fasal Bima Yojana (PMFBY) is the flagship national Crop Insurance scheme. It provides "
                        "comprehensive financial coverage against non-preventable natural risks (drought, flood, pests, post-harvest losses). "
                        "Farmers pay a uniform actuarial premium of only 2% for all Kharif food and oilseed crops, 1.5% for all Rabi crops, "
                        "and 5% for commercial/annual horticultural crops, with the balance shared equally by the Central and State Governments. "
                        "In Tamil Nadu, the state co-administers PMFBY to disburse claims directly to farmers' bank accounts following crop cutting experiments."
                    ),
                    "metadata": {
                        "source": "pmfby.gov.in (Official Crop Insurance Guidelines)",
                        "scheme_name": "Crop Insurance (PMFBY)",
                        "level": "Central",
                        "category": "insurance"
                    }
                },
                {
                    "text": (
                        "Kalaignar All Village Integrated Agriculture Development Programme (Kalaignarin Anaithu Grama Orunginaintha "
                        "Velan Valarchi Thittam) is a flagship scheme implemented by the Government of Tamil Nadu. The scheme aims to "
                        "bring fallow lands into active cultivation, augment water resources through farm ponds and check dams, distribute "
                        "coconut seedlings, power sprayers, and horticulture saplings at subsidized rates, and convert dry lands to irrigated "
                        "farmlands across all Village Panchayats in Tamil Nadu."
                    ),
                    "metadata": {
                        "source": "tn.gov.in/department/agriculture",
                        "scheme_name": "Kalaignar Scheme (TN)",
                        "level": "Tamil Nadu",
                        "category": "income_support"
                    }
                }
            ]
            engine.ingest_documents(baseline_docs)
            st.sidebar.success("Successfully indexed all 4 baseline schemes!")
            st.rerun()

    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.last_processed_audio_hash = None
        st.session_state.simulated_voice_prompt = None
        st.session_state.current_pdf_id = None
        engine.memory.clear()
        st.rerun()

    return {
        "scheme_name": None if scheme_filter == "All" else scheme_filter,
        "level": None if level_filter == "All" else level_filter,
        "category": None if category_filter == "All" else category_filter
    }


def main():
    init_session_state()
    engine = get_rag_engine()
    filters = render_sidebar(engine)

    st.markdown('<p class="main-header">🌾 Bilingual Agriculture Scheme Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Official assistance for Central & Tamil Nadu Agricultural Welfare Schemes</p>', unsafe_allow_html=True)

    # Display Chat History
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            else:
                data: BilingualSchemeResponse = msg["data"]
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**English Explanation**")
                    st.write(data.english_response)
                with col2:
                    st.markdown("**தமிழ் விளக்கம்**")
                    st.markdown(f"<div class='tamil-text'>{data.tamil_response}</div>", unsafe_allow_html=True)
                    
                    if data.is_agricultural_query:
                        if st.button("🔊 Listen in Tamil (கேளுங்கள்)", key=f"audio_btn_{idx}"):
                            audio_bytes = generate_tamil_audio(data.tamil_response)
                            st.audio(audio_bytes, format="audio/mp3")

                with st.expander("🔍 RAG Grounding & Confidence Verification"):
                    st.metric("Retrieval Grounding Confidence", f"{int(data.confidence_score * 100)}%")
                    st.write(f"**Verified Citations:** {', '.join(data.citations) if data.citations else 'None'}")
                    st.caption("Score is derived from vector embedding cosine similarity across indexed government gazettes.")

    # Voice Recording Input & Simulation Fallback
    col_mic, col_sim = st.columns([1, 1])
    with col_mic:
        voice_audio = st.audio_input("🎙️ Hardware Mic Input")
    with col_sim:
        st.caption("No working microphone? Test voice simulation:")
        if st.button("🧪 Simulate Farmer Voice Query: 'PM Kisan scheme details'"):
            st.session_state.simulated_voice_prompt = "Tell me about PM Kisan scheme eligibility and financial assistance"
            st.rerun()

    query_to_execute = None

    # Process Hardware Voice Audio
    if voice_audio is not None:
        audio_bytes = voice_audio.getvalue()
        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        if audio_hash != st.session_state.last_processed_audio_hash:
            with st.spinner("Transcribing speech via Whisper..."):
                query_to_execute = transcribe_audio(voice_audio)
                st.session_state.last_processed_audio_hash = audio_hash

    # Process Simulated Voice Audio
    if st.session_state.simulated_voice_prompt:
        query_to_execute = st.session_state.simulated_voice_prompt
        st.session_state.simulated_voice_prompt = None

    # Standard Chat Input Box
    chat_input_val = st.chat_input("Ask scheme questions (e.g., 'Who is eligible for PM Kisan and how much is given?')")
    if chat_input_val:
        query_to_execute = chat_input_val

    # Execute RAG Query Pipeline strictly on intentional input
    if query_to_execute:
        st.session_state.messages.append({"role": "user", "content": query_to_execute})
        with st.chat_message("user"):
            st.write(query_to_execute)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving verified gazette details & translating..."):
                response: BilingualSchemeResponse = engine.query(
                    question=query_to_execute,
                    filter_metadata=filters
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**English Explanation**")
                    st.write(response.english_response)
                with col2:
                    st.markdown("**தமிழ் விளக்கம்**")
                    st.markdown(f"<div class='tamil-text'>{response.tamil_response}</div>", unsafe_allow_html=True)
                    
                    if response.is_agricultural_query:
                        audio_stream = generate_tamil_audio(response.tamil_response)
                        st.audio(audio_stream, format="audio/mp3")

                with st.expander("🔍 RAG Grounding & Confidence Verification", expanded=True):
                    st.metric("Retrieval Grounding Confidence", f"{int(response.confidence_score * 100)}%")
                    st.write(f"**Verified Citations:** {', '.join(response.citations)}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.english_response,
                    "data": response
                })


if __name__ == "__main__":
    main()