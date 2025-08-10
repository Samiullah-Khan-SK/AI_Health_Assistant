import streamlit as st
from langchain.schema import HumanMessage
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.tools.tavily_search import TavilySearchResults
from fpdf import FPDF
from datetime import datetime
import re
import PyPDF2
from io import BytesIO
import os
import string

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Health Assistant", layout="wide", page_icon="🩺")
st.title("🩺 AI Health Assistant")
st.markdown("Ask about **symptoms**, **diseases**, **medicines**, or **general health advice**.")

# Add WhatsApp-like styling
st.markdown("""
<style>
    /* Modern gradient background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    }
    
    /* Elegant sidebar */
    [data-testid=stSidebar] {
        background: white !important;
        box-shadow: 2px 0 15px rgba(0,0,0,0.05);
        border-right: none !important;
    }
    
    /* Professional chat bubbles */
    .user-message {
        background: linear-gradient(135deg, #6e8efb 0%, #4a6cf7 100%);
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 18px;
        margin: 8px 0;
        max-width: 78%;
        float: right;
        box-shadow: 0 4px 12px rgba(74, 108, 247, 0.2);
        line-height: 1.5;
        font-size: 15px;
        border: none;
    }
    
    .bot-message {
        background: white;
        color: #333;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 18px;
        margin: 8px 0;
        max-width: 78%;
        float: left;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        line-height: 1.5;
        font-size: 15px;
        border: 1px solid #f0f0f0;
    }
    
    /* Message container - ADDED BOTTOM PADDING */
    .message-container {
        display: flex;
        flex-direction: column;
        background: transparent;
        padding: 0 10px 100px; /* Added bottom padding */
    }
    
    /* Avatars */
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        margin: 0 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        flex-shrink: 0;
    }
    
    .user-avatar {
        background: linear-gradient(135deg, #4a6cf7 0%, #2541f5 100%);
        color: white;
        font-weight: bold;
    }
    
    .bot-avatar {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
        color: #4a6cf7;
        font-weight: bold;
        border: 2px solid white;
    }
    
    /* Timestamp - ADDED Z-INDEX FIX */
    .timestamp {
        font-size: 0.65em;
        color: #a8b0c0;
        margin-top: 3px;
        font-weight: 500;
        position: relative; /* Ensures z-index works */
        z-index: 1; /* Keeps timestamps below input */
    }
    
    /* Chat bubble alignment */
    .chat-bubble {
        display: flex;
        margin: 10px 0;
        transition: all 0.3s ease;
        position: relative; /* Ensures stacking context */
        z-index: auto; /* Default stacking */
    }
    
    .chat-bubble:hover {
        transform: translateY(-1px);
    }
    
    /* Input area - FIXED OVERLAP ISSUE */
    .stChatFloatingInputContainer {
        background: rgba(255,255,255,0.95) !important; /* More opaque */
        backdrop-filter: blur(8px); /* Stronger blur */
        border-top: 1px solid #f0f0f0;
        position: relative; /* Required for z-index */
        z-index: 1000; /* Ensures input stays on top */
        padding: 15px 0; /* Added vertical spacing */
    }
    
    .stChatInputContainer {
        background: white !important;
        border-radius: 24px !important;
        padding: 10px 16px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important; /* Stronger shadow */
        border: 1px solid #e4e8f0 !important;
        position: relative; /* Creates stacking context */
        z-index: 1001; /* Higher than container */
    }
    
    .stChatInputContainer textarea {
        background: transparent !important;
        color: #333 !important;
        font-size: 15px !important;
        padding: 10px !important;
    }
    
    .stChatInputContainer textarea::placeholder {
        color: #a8b0c0 !important;
    }
    
    .stChatInputContainer textarea:focus {
        box-shadow: none !important;
        border-color: #4a6cf7 !important;
    }
    
    .stChatInputContainer button {
        background: linear-gradient(135deg, #6e8efb 0%, #4a6cf7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        box-shadow: 0 4px 10px rgba(74, 108, 247, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    
    .stChatInputContainer button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 15px rgba(74, 108, 247, 0.5) !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f5f7fa;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4a6cf7;
        border-radius: 3px;
    }
    
    /* Markdown content styling */
    .bot-message strong {
        color: #4a6cf7;
    }
    
    .bot-message a {
        color: #4a6cf7;
        text-decoration: none;
        font-weight: 500;
    }
    
    .bot-message ul, 
    .bot-message ol {
        padding-left: 20px;
        margin: 8px 0;
    }
    
    .bot-message li {
        margin-bottom: 6px;
    }
    
    /* Animation for new messages */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .chat-bubble {
        animation: fadeIn 0.3s ease-out;
    }
    
    /* NEW: Prevent content overflow behind input */
    .stChatMessageContainer {
        padding-bottom: 100px !important;
    }
</style>
""", unsafe_allow_html=True)
# --- CONSTANTS ---
GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "howdy", "hola", "greetings"}

HEALTH_KEYWORDS = {
    # Diseases/Conditions
    "disease", "illness", "condition", "disorder", "syndrome", "infection", "chronic", "acute",
    "cancer", "tumor", "diabetes", "hypertension", "asthma", "arthritis", "glaucoma", "depression", 
    "anxiety", "alzheimer", "parkinson", "stroke", "heart attack", "pneumonia", "bronchitis",
    "osteoporosis", "epilepsy", "migraine", "autism", "adhd", "bipolar", "schizophrenia",
    
    # Symptoms
    "symptom", "sign", "indication", "pain", "ache", "fever", "headache", "nausea", "vomit",
    "dizziness", "fatigue", "weakness", "rash", "swelling", "inflammation", "bleeding",
    
    # Treatments
    "treatment", "therapy", "medication", "medicine", "drug", "prescription", "surgery", 
    "operation", "rehabilitation", "recovery", "physical therapy", "chemotherapy", "radiation",
    "vaccine", "immunization", "antibiotic", "antiviral", "painkiller", "analgesic",
    
    # Body Parts/Systems
    "heart", "lung", "liver", "kidney", "brain", "stomach", "intestine", "muscle", "bone",
    "joint", "nerve", "blood", "immune", "respiratory", "digestive", "nervous", "endocrine",
    
    # General Health
    "health", "wellness", "prevention", "diagnosis", "prognosis", "checkup", "screening",
    "test", "scan", "x-ray", "mri", "ct", "ultrasound", "biopsy", "blood test", "urine test",
    
    # Medications
    "dosage", "side effect", "interaction", "contraindication", "overdose", "allergy",
    "tablet", "capsule", "injection", "ointment", "cream", "syrup", "drops", "suppository",
    
    # Specialties
    "cardiology", "neurology", "oncology", "pediatrics", "psychiatry", "dermatology",
    "endocrinology", "gastroenterology", "hematology", "nephrology", "pulmonology",
    
    # Report Related
    "report", "result", "finding", "diagnosis", "medical report", "lab report", "scan report",
    "blood report", "urine report", "imaging report", "pathology report"
}

MEDICATION_QUESTIONS = [
    "What are the side effects of {topic}?",
    "What is the recommended dosage for {topic}?",
    "What should I do if I miss a dose of {topic}?"
]

CONDITION_QUESTIONS = [
    "What are the main symptoms of {topic}?",
    "How is {topic} diagnosed?",
    "What are the treatment options for {topic}?",
    "Can {topic} be prevented?",
    "What are the complications of {topic}?"
]

# --- HELPER FUNCTIONS ---
def is_greeting(message):
    message_lower = message.lower().strip()
    return message_lower in GREETINGS or any(message_lower.startswith(g + ' ') for g in GREETINGS)

def is_health_related(message):
    message_lower = message.lower()
    
    if st.session_state.pdf_text and ("report" in message_lower or "result" in message_lower or "diagnosis" in message_lower):
        return True
        
    if any(q.lower().format(topic=".*") in message_lower for q in MEDICATION_QUESTIONS + CONDITION_QUESTIONS):
        return True
        
    keyword_match = any(keyword in message_lower for keyword in HEALTH_KEYWORDS)
    
    pattern_match = bool(re.search(
        r'\b(what is|define|explain|symptoms of|treatment for|causes of|how to treat|signs of|how can i|how long does|what are the|can [a-z]+ be)\b', 
        message_lower
    ))
    
    medication_match = bool(re.search(
        r'\b(side effects|dosage|drug interactions|take with food|how long to work|recommended dose)\b',
        message_lower
    ))
    
    return keyword_match or pattern_match or medication_match

def detect_intent(question):
    q = question.lower().strip()
    
    # Check for simple definition requests first (short questions)
    if (any(k in q for k in ["what is", "define", "definition"]) 
        and len(q.split()) <= 6):  # Simple questions like "what is X"
        return "simple_definition"
    
    # Check for report analysis
    if any(term in q for term in ["report", "result", "diagnosis", "lab test", "scan"]):
        return "report_analysis"
        
    # Check for personal case (with age)
    if (re.search(r'\b(i am|i have|my|me)\b', q) 
        and re.search(r'\d+\s*(year|yr)s? old', q)):
        return "personal_case"
        
    # Check for symptom queries
    if any(k in q for k in ["symptom", "sign", "indication", "manifestation"]):
        return "symptoms"
        
    # Check for treatment queries
    if any(k in q for k in ["treatment", "cure", "remedy", "therapy", "manage", "management"]):
        return "treatment"
        
    # Check for medication/drug queries
    if any(k in q for k in ["medication", "drug", "medicine", "pill", "tablet", "dosage"]):
        return "medication"
        
    # Check for pain/emergency cases
    if (re.search(r'\b(i|my|me)\b', q) 
        and any(k in q for k in ["pain", "ache", "hurt", "swelling", "bleeding", "emergency"])):
        return "personal_case"
        
    # Check for definition requests
    if any(k in q for k in ["what is", "define", "explain", "meaning of"]):
        return "definition"
        
    return "general"

def extract_age_symptoms_format(user_question, ai_response):
    age_match = re.search(r'(\d{1,3})\s*(year|yr)[s]*\s*old', user_question, re.IGNORECASE)
    symptoms = re.findall(r'(?:have|has|having|with|experienced)\s+([^.,?]+)', user_question, re.IGNORECASE)
    symptoms = [s.strip() for s in symptoms if s.strip()]
    if not symptoms:
        return ai_response
    bullets = "\n".join([f"- {s.capitalize()}" for s in symptoms])
    if age_match:
        return f"**{age_match.group(1)}-year-old patient presenting with:**\n{bullets}\n\n**Recommendations:**\n{ai_response}"
    return f"**Patient presenting with:**\n{bullets}\n\n**Recommendations:**\n{ai_response}"

def get_trusted_domains(query):
    core_domains = ["mayoclinic.org", "webmd.com", "medlineplus.gov", 
                   "cdc.gov", "who.int", "drugs.com"]
    
    query_lower = query.lower()
    
    # Add niche sources based on query
    if "cancer" in query_lower:
        core_domains.append("cancer.org")
    if "mental" in query_lower or "depression" in query_lower or "anxiety" in query_lower:
        core_domains.append("nimh.nih.gov")
    if "child" in query_lower or "pediatric" in query_lower:
        core_domains.append("healthychildren.org")
    if "alternative" in query_lower or "herbal" in query_lower:
        core_domains.append("nccih.nih.gov")
    
    return core_domains

def generate_follow_ups(question):
    """Generate sensible medical follow-up questions only when appropriate"""
    # List of medical conditions we can safely generate follow-ups for
    MEDICAL_CONDITIONS = {
    # Cardiovascular
    "hypertension": ["high blood pressure", "hbp", "bp"],
    "heart attack": ["myocardial infarction", "mi", "cardiac arrest"],
    "arrhythmia": ["irregular heartbeat", "afib", "atrial fibrillation"],
    "coronary artery disease": ["cad", "heart disease", "artery blockage"],
    
    # Endocrine
    "diabetes": ["blood sugar", "diabetic", "type 1 diabetes", "type 2 diabetes"],
    "hypothyroidism": ["underactive thyroid", "low thyroid", "thyroid disorder"],
    "hyperthyroidism": ["overactive thyroid", "graves disease"],
    
    # Respiratory
    "asthma": ["breathing difficulty", "wheezing", "bronchial"],
    "copd": ["chronic obstructive pulmonary disease", "emphysema", "chronic bronchitis"],
    "pneumonia": ["lung infection", "respiratory infection"],
    
    # Neurological
    "migraine": ["headache", "severe headache", "ocular migraine"],
    "epilepsy": ["seizures", "convulsions"],
    "parkinson's disease": ["tremors", "shaking palsy"],
    "alzheimer's disease": ["dementia", "memory loss"],
    
    # Gastrointestinal
    "gerd": ["acid reflux", "heartburn", "gastroesophageal reflux"],
    "ibs": ["irritable bowel syndrome", "digestive issues", "bowel disorder"],
    "crohn's disease": ["inflammatory bowel disease", "ibd"],
    
    # Musculoskeletal
    "arthritis": ["joint pain", "rheumatoid arthritis", "osteoarthritis"],
    "osteoporosis": ["bone loss", "weak bones"],
    "back pain": ["lumbago", "sciatica", "herniated disc"],
    
    # Mental Health
    "depression": ["major depressive disorder", "mdd", "clinical depression"],
    "anxiety": ["anxiety disorder", "panic attacks", "generalized anxiety"],
    "adhd": ["attention deficit disorder", "add", "hyperactivity"],
    
    # Infectious Diseases
    "covid-19": ["coronavirus", "sars-cov-2"],
    "influenza": ["flu", "seasonal flu"],
    "urinary tract infection": ["uti", "bladder infection"],
    
    # Dermatological
    "eczema": ["atopic dermatitis", "skin rash"],
    "psoriasis": ["scaly skin", "plaques"],
    "acne": ["pimples", "zits", "breakouts"],
    
    # Cancer/Tumors
    "breast cancer": ["mammary carcinoma", "breast tumor"],
    "lung cancer": ["pulmonary carcinoma", "smoking cancer"],
    "melanoma": ["skin cancer", "malignant mole"],
    
    # Women's Health
    "endometriosis": ["pelvic pain", "menstrual pain"],
    "pcos": ["polycystic ovary syndrome", "hormonal imbalance"],
    "menopause": ["hot flashes", "perimenopause"],
    
    # Men's Health
    "bph": ["benign prostatic hyperplasia", "enlarged prostate"],
    "erectile dysfunction": ["ed", "impotence"],
    
    # Pediatric
    "adhd": ["attention deficit hyperactivity disorder"],
    "autism": ["asd", "autism spectrum disorder"],
    
    # Eye Conditions
    "glaucoma": ["eye pressure", "optic nerve damage"],
    "cataracts": ["cloudy vision", "lens opacity"],
    
    # Blood Disorders
    "anemia": ["low hemoglobin", "iron deficiency"],
    "leukemia": ["blood cancer", "white blood cell disorder"],
    
    # Autoimmune
    "lupus": ["sle", "systemic lupus erythematosus"],
    "rheumatoid arthritis": ["joint inflammation", "autoimmune arthritis"],
    
    # Chronic Conditions
    "chronic kidney disease": ["ckd", "renal failure"],
    "chronic pain": ["persistent pain", "pain syndrome"],
    
    # Additions based on your previous needs
    "medulloblastoma": ["brain tumor", "pediatric brain cancer"],
    "lower back pain": ["lumbar pain", "sciatica"],
    "high cholesterol": ["hyperlipidemia", "ldl", "hdl"]
    }
    
    # Clean the question
    question_lower = question.lower().translate(str.maketrans("", "", string.punctuation))
    
    # Check for report analysis first
    if "report" in question_lower and st.session_state.pdf_text:
        conditions = re.findall(r'(?:diagnos|assess|find).*?([A-Z][a-zA-Z\s-]+(?:disease|syndrome|disorder|condition))', 
                             st.session_state.pdf_text, re.IGNORECASE)
        if conditions:
            topic = conditions[0].strip()
            topic = re.sub(r'\s{2,}', ' ', topic)
            topic = topic.split(',')[0]
            topic = topic.split(' and ')[0]
            return [
                f"What are the symptoms of {topic}?",
                f"How is {topic} treated?",
                f"What causes {topic}?"
            ]
    
    # Check for known medical conditions
    for condition, keywords in MEDICAL_CONDITIONS.items():
        if (condition in question_lower or 
            any(kw in question_lower for kw in keywords)):
            return [
                f"What are the warning signs of {condition}?",
                f"How is {condition} diagnosed?",
                f"What are the best treatments for {condition}?",
                f"Can {condition} be prevented?"
            ]
    
    # Return empty list if no good follow-ups found
    return []

def generate_prompt(intent, context, question):
    # Extract and clean the main topic
    topic = re.sub(r'[^\w\s]', '', question)
    topic = ' '.join([word for word in topic.split() 
                     if word.lower() not in ["what", "is", "are", "the", "define", "explain"]])
    topic = topic.strip()
    
    # Base formatting template
    format_instructions = """
    Response Format Requirements:
    ✦ Use emoji bullet points (✦ for main points, ▪ for details)
    ✦ Bold all medical terms (**term**)
    ✦ Section headers with ▷ symbol
    ✦ Maximum 3 statistics if available
    ✦ Clear visual separation between sections
    ✦ Sources at bottom with 📚 emoji
    """
    
    if intent == "simple_definition":
        return f"""Provide a 1-2 sentence definition of **{topic}** formatted as:
        ✦ **Definition:** [concise explanation]
        ✦ **Significance:** [1 sentence importance]
        {format_instructions}"""
        
    if intent == "report_analysis":
        return f"""Analyze this medical report:
        Report Content: {context}
        Patient Question: {question}
        
        Required Structure:
        ▷ **Key Findings** (3-5 bullet points)
        ▷ **Explanation** (simple terms)
        ▷ **Recommended Actions** (prioritized)
        ▷ **Urgency Level** (⚠️ symbols if needed)
        {format_instructions}"""
    
    intent_specific = {
        "definition": f"""
        ▷ **Definition of {topic}:**
        ✦ Core concept (2 sentences max)
        ✦ Key characteristics
        
        ▷ **Clinical Significance:**
        ✦ Why it matters
        ✦ 1-2 prevalence stats if available""",
        
        "symptoms": f"""
        ▷ **Symptoms of {topic}:**
        ✦ Most common manifestations
        ▪ Prevalence percentages if known
        ✦ Red flags (use ⚠️)
        
        ▷ **When to Seek Help:**
        ✦ Warning signs""",
        
        "treatment": f"""
        ▷ **Treatment Options for {topic}:**
        ✦ **Medications:**
        ▪ First-line drugs
        ▪ Common side effects
        
        ✦ **Therapies:**
        ▪ Evidence-based approaches
        
        ✦ **Lifestyle Modifications:**
        ▪ Most effective changes""",
        
        "prevention": f"""
        ▷ **Preventing {topic}:**
        ✦ Proactive strategies (emoji bullets)
        ✦ Risk reduction tips
        
        ▷ **Early Detection:**
        ✦ Screening recommendations""",
        
        "medication": f"""
        ▷ **About {topic}:**
        ✦ **Dosage:** Standard ranges
        ✦ **Side Effects:** Most common
        ✦ **Interactions:** Key warnings
        ✦ **Precautions:** Special populations""",
        
        "general": f"""
        ▷ **About {topic}:**
        ✦ Key facts
        ✦ Important considerations
        
        ▷ **Quick Facts:**
        ▪ Prevalence
        ▪ Risk factors"""
    }
    
    return f"""Generate a professional medical response about **{topic}** with:
    Question: {question}
    Context: {context}
    
    {intent_specific.get(intent, 'Provide accurate, formatted information')}
    {format_instructions}
    
    Additional Guidelines:
    - Use **bold** for first topic mention and medical terms
    - Keep paragraphs under 3 lines
    - Use ➡️ for progression arrows when needed
    - Add ❗ for important warnings"""

def load_faiss_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.exists("vectorstore_faiss"):
        # Create an empty FAISS store if it doesn't exist
        return FAISS.from_texts(["Initial empty document"], embeddings)
    return FAISS.load_local("vectorstore_faiss", embeddings, allow_dangerous_deserialization=True)

def extract_pdf_text(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = "\n".join([p.extract_text() or '' for p in reader.pages])
        # Clean up the text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Replace single newlines with space
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def save_chat_history(history):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="AI Health Assistant - Chat Transcript", ln=True, align='C')
    pdf.ln(10)
    for h in history:
        user = h["user"].encode("latin-1", "replace").decode("latin-1")
        bot = h["bot"].encode("latin-1", "replace").decode("latin-1")
        pdf.set_font("Arial", 'B', 12)
        pdf.multi_cell(0, 10, f"You: {user}")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"Assistant: {bot}")
        pdf.ln(5)
        pdf.cell(0, 10, "-" * 80, ln=True)
    return pdf.output(dest="S").encode("latin-1", "replace")

# --- INITIALIZATION ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        st.session_state.vector_store = load_faiss_store()
if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "processing" not in st.session_state:
    st.session_state.processing = False
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# --- MAIN CHAT DISPLAY ---
chat_container = st.container()
with chat_container:
    st.markdown("<div class='message-container'>", unsafe_allow_html=True)
    
    for chat in st.session_state.chat_history:
        # User message
        st.markdown(f"""
        <div class='chat-bubble'>
            <div class='user-container'>
                <div style='display: flex; align-items: center; justify-content: flex-end; width: 100%;'>
                    <div class='user-message'>{chat["user"]}</div>
                    <div class='avatar user-avatar'>👤</div>
                </div>
                <div class='timestamp' style='text-align: right;'>{chat["timestamp"]}</div>
            </div>
        </div>
        <div class='clear'></div>
        """, unsafe_allow_html=True)
        
        # Bot message
        if chat["bot"] != "Thinking...":
            st.markdown(f"""
            <div class='chat-bubble'>
                <div class='bot-container'>
                    <div style='display: flex; align-items: center; width: 100%;'>
                        <div class='avatar bot-avatar'>🩺</div>
                        <div class='bot-message'>{chat["bot"]}</div>
                    </div>
                    <div class='timestamp'>{chat["timestamp"]}</div>
                </div>
            </div>
            <div class='clear'></div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("📄 Upload Medical Report")
    uploaded_pdf = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
    if uploaded_pdf and not st.session_state.processing and not st.session_state.pdf_processed:
        st.session_state.processing = True
        with st.spinner("Processing PDF..."):
            try:
                pdf_text = extract_pdf_text(uploaded_pdf)
                if pdf_text:
                    st.session_state.pdf_text = pdf_text
                    # Create a new FAISS store with the PDF content
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    st.session_state.vector_store = FAISS.from_texts([pdf_text], embeddings)
                    st.success("PDF processed successfully!")
                    st.session_state.pdf_processed = True
                else:
                    st.error("Failed to extract text from PDF")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
            finally:
                st.session_state.processing = False
                st.rerun()
    
    st.markdown("---")
    if st.button("📥 Download Chat as PDF", use_container_width=True):
        if st.session_state.chat_history:
            pdf_bytes = save_chat_history(st.session_state.chat_history)
            fname = f"health_chat_{st.session_state.session_start}.pdf"
            st.download_button("💾 Save PDF", pdf_bytes, file_name=fname, mime="application/pdf")
        else:
            st.warning("No chat history to download.")
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    if st.button("🔄 Reset PDF", use_container_width=True):
        st.session_state.pdf_text = ""
        st.session_state.pdf_processed = False
        st.session_state.vector_store = load_faiss_store()
        st.rerun()

# --- CHAT PROCESSING ---
user_query = st.chat_input("Ask a health question...")
if user_query and not st.session_state.processing:
    # Add user message to history immediately
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append({
        "user": user_query,
        "bot": "Thinking...",
        "timestamp": timestamp
    })
    st.session_state.processing = True
    st.rerun()

if st.session_state.processing and st.session_state.chat_history and st.session_state.chat_history[-1]["bot"] == "Thinking...":
    user_query = st.session_state.chat_history[-1]["user"]
    
    try:
        if is_greeting(user_query):
            bot_reply = "Hello! 👋 I'm your AI Health Assistant. How can I help you with medical questions today?"
        elif not is_health_related(user_query):
            bot_reply = "I'm sorry, I can only assist with health-related questions."
        else:
            intent = detect_intent(user_query)
            vector_store = st.session_state.vector_store
            
            try:
                if st.session_state.pdf_text:
                    pdf_results = vector_store.similarity_search(user_query, k=3)
                    pdf_context = "\n\n".join([r.page_content for r in pdf_results])
                else:
                    pdf_context = ""
                
                general_results = vector_store.similarity_search(user_query, k=3)
                general_context = "\n\n".join([r.page_content for r in general_results])
                
                if pdf_context:
                    context = f"PATIENT RECORD:\n{pdf_context}\n\nMEDICAL KNOWLEDGE:\n{general_context}"
                else:
                    context = general_context
                
                if intent == "report_analysis":
                    if not st.session_state.pdf_text:
                        bot_reply = "Please upload a medical report PDF first."
                    else:
                        context = f"MEDICAL REPORT:\n{st.session_state.pdf_text}\n\nCONTEXT:\n{general_context}"
                
                fallback = not context.strip() or len(context.strip()) < 50
                
                try:
                    groq_chat = ChatOpenAI(
                        openai_api_key=st.secrets.get("GROQ_API_KEY", ""),
                        base_url="https://api.groq.com/openai/v1",
                        model_name="llama3-70b-8192",
                        temperature=0.3,
                        request_timeout=30
                    )
                    
                    if fallback:
                        try:
                            tavily = TavilySearchResults(api_key=st.secrets.get("TAVILY_API_KEY", ""))
                            trusted_domains = get_trusted_domains(user_query)
                            search_results = tavily.run({
                                "query": user_query,
                                "include_domains": trusted_domains,
                                "num_results": 5 if intent in ["report_analysis", "personal_case"] else 3
                            })
                            context = "\n\n".join([r["content"] for r in search_results])
                            st.session_state.vector_store.add_texts([context])
                        except Exception as e:
                            st.error(f"Search error: {str(e)}")
                            context = "Current medical knowledge"
                    
                    prompt = generate_prompt(intent, context, user_query)
                    response = groq_chat([HumanMessage(content=prompt)])
                    ai_reply = response.content
                    
                    if intent == "personal_case":
                        bot_reply = extract_age_symptoms_format(user_query, ai_reply)
                    else:
                        bot_reply = ai_reply
                    
                    if not is_greeting(user_query):
                        followups = generate_follow_ups(user_query)
                        if followups:
                            bot_reply += "\n\n---\n**You might also ask:**\n" + "\n".join([f"• {q}" for q in followups])
                    
                except Exception as e:
                    bot_reply = f"⚠️ Error processing request: {str(e)}"
            except Exception as e:
                bot_reply = f"⚠️ Search error: {str(e)}"
        
        st.session_state.chat_history[-1]["bot"] = bot_reply
        st.session_state.processing = False
        st.rerun()
        
    except Exception as e:
        st.session_state.chat_history[-1]["bot"] = f"⚠️ Error: {str(e)}"
        st.session_state.processing = False
        st.rerun()

# --- PDF SUMMARY ---
if st.session_state.pdf_text:
    with st.expander("📄 Medical Report Summary"):
        st.subheader("Extracted Medical Information")
        st.text(st.session_state.pdf_text[:1500] + ("..." if len(st.session_state.pdf_text) > 1500 else ""))
