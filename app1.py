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
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    }
    
    [data-testid=stSidebar] {
        background: white !important;
        box-shadow: 2px 0 15px rgba(0,0,0,0.05);
    }
    
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
    
    .message-container {
        display: flex;
        flex-direction: column;
        background: transparent;
        padding: 0 10px 100px;
    }
    
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
    
    .timestamp {
        font-size: 0.65em;
        color: #a8b0c0;
        margin-top: 3px;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }
    
    .chat-bubble {
        display: flex;
        margin: 10px 0;
        transition: all 0.3s ease;
        position: relative;
        z-index: auto;
    }
    
    .stChatFloatingInputContainer {
        background: rgba(255,255,255,0.95) !important;
        backdrop-filter: blur(8px);
        border-top: 1px solid #f0f0f0;
        position: relative;
        z-index: 1000;
        padding: 15px 0;
    }
    
    .stChatInputContainer {
        background: white !important;
        border-radius: 24px !important;
        padding: 10px 16px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
        border: 1px solid #e4e8f0 !important;
        position: relative;
        z-index: 1001;
    }
    
    .stChatInputContainer textarea {
        background: transparent !important;
        color: #333 !important;
        font-size: 15px !important;
        padding: 10px !important;
    }
    
    .stChatInputContainer button {
        background: linear-gradient(135deg, #6e8efb 0%, #4a6cf7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
    }
    
    .stChatMessageContainer {
        padding-bottom: 100px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "howdy", "hola", "greetings"}

HEALTH_KEYWORDS = {
    "disease", "illness", "condition", "disorder", "syndrome", "infection", "chronic", "acute",
    "cancer", "tumor", "diabetes", "hypertension", "asthma", "arthritis", "glaucoma", "depression", 
    "anxiety", "alzheimer", "parkinson", "stroke", "heart attack", "pneumonia", "bronchitis",
    "osteoporosis", "epilepsy", "migraine", "autism", "adhd", "bipolar", "schizophrenia",
    "symptom", "sign", "indication", "pain", "ache", "fever", "headache", "nausea", "vomit",
    "dizziness", "fatigue", "weakness", "rash", "swelling", "inflammation", "bleeding",
    "treatment", "therapy", "medication", "medicine", "drug", "prescription", "surgery", 
    "operation", "rehabilitation", "recovery", "physical therapy", "chemotherapy", "radiation",
    "vaccine", "immunization", "antibiotic", "antiviral", "painkiller", "analgesic",
    "heart", "lung", "liver", "kidney", "brain", "stomach", "intestine", "muscle", "bone",
    "joint", "nerve", "blood", "immune", "respiratory", "digestive", "nervous", "endocrine",
    "health", "wellness", "prevention", "diagnosis", "prognosis", "checkup", "screening",
    "test", "scan", "x-ray", "mri", "ct", "ultrasound", "biopsy", "blood test", "urine test",
    "dosage", "side effect", "interaction", "contraindication", "overdose", "allergy",
    "tablet", "capsule", "injection", "ointment", "cream", "syrup", "drops", "suppository",
    "cardiology", "neurology", "oncology", "pediatrics", "psychiatry", "dermatology",
    "endocrinology", "gastroenterology", "hematology", "nephrology", "pulmonology",
    "report", "result", "finding", "diagnosis", "medical report", "lab report", "scan report"
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
    
    # Always allow personal health descriptions
    if re.search(r'\b(i am|i have|my|me|father|mother|child|husband|wife|son|daughter)\b', message_lower):
        if any(symptom in message_lower for symptom in ['headache', 'fever', 'pain', 'cough', 'sore throat', 'confusion', 'dizziness', 'nausea', 'vomiting', 'rash']):
            return True
    
    # Check for report analysis
    if st.session_state.pdf_text and ("report" in message_lower or "result" in message_lower or "diagnosis" in message_lower):
        return True
        
    # Check for follow-up questions
    if any(q.lower().format(topic=".*") in message_lower for q in MEDICATION_QUESTIONS + CONDITION_QUESTIONS):
        return True
        
    # Check for health keywords
    keyword_match = any(keyword in message_lower for keyword in HEALTH_KEYWORDS)
    
    # Check for question patterns
    pattern_match = bool(re.search(
        r'\b(what is|define|explain|symptoms of|treatment for|causes of|how to treat|signs of|how can i|how long does|what are the|can [a-z]+ be)\b', 
        message_lower
    ))
    
    # Check for medication patterns
    medication_match = bool(re.search(
        r'\b(side effects|dosage|drug interactions|take with food|how long to work|recommended dose)\b',
        message_lower
    ))
    
    # Check for personal health descriptions
    personal_health = bool(re.search(
        r'\b(\d+\s*(year|yr)s?\s*old|age\s*\d+)\b.*\b(with|has|having|experienced)\b',
        message_lower
    ))
    
    return keyword_match or pattern_match or medication_match or personal_health

def detect_intent(question):
    q = question.lower().strip()
    
    # Check for report analysis first
    if any(term in q for term in ["report", "result", "diagnosis", "lab test", "scan"]):
        return "report_analysis"
    
    # Check for personal case (with age and symptoms)
    if (re.search(r'\b(i am|i have|my|me|father|mother|child|husband|wife)\b', q) 
        and (re.search(r'\d+\s*(year|yr)s? old', q) or any(symptom in q for symptom in ['pain', 'fever', 'headache', 'cough', 'sore throat']))):
        return "personal_case"
    
    # Check for medication queries
    if any(k in q for k in ["medication", "drug", "medicine", "pill", "tablet", "dosage", "side effect"]):
        return "medication"
    
    # Check for symptom queries
    if any(k in q for k in ["symptom", "sign", "indication", "manifestation"]):
        return "symptoms"
    
    # Check for treatment queries
    if any(k in q for k in ["treatment", "cure", "remedy", "therapy", "manage", "management"]):
        return "treatment"
    
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
    
    if "cancer" in query_lower:
        core_domains.append("cancer.org")
    if "mental" in query_lower or "depression" in query_lower or "anxiety" in query_lower:
        core_domains.append("nimh.nih.gov")
    if "child" in query_lower or "pediatric" in query_lower:
        core_domains.append("healthychildren.org")
    
    return core_domains

def generate_follow_ups(question):
    """Generate sensible medical follow-up questions only when appropriate"""
    question_lower = question.lower()
    
    # Check for report analysis first
    if "report" in question_lower and st.session_state.pdf_text:
        conditions = re.findall(r'(?:diagnos|assess|find).*?([A-Z][a-zA-Z\s-]+(?:disease|syndrome|disorder|condition))', 
                             st.session_state.pdf_text, re.IGNORECASE)
        if conditions:
            topic = conditions[0].strip()
            topic = re.sub(r'\s{2,}', ' ', topic)
            topic = topic.split(',')[0]
            return [
                f"What are the symptoms of {topic}?",
                f"How is {topic} treated?",
                f"What causes {topic}?"
            ]
    
    # Common medical conditions for follow-ups
    medical_conditions = {
        "headache": "headache", "fever": "fever", "cough": "cough", 
        "sore throat": "sore throat", "pain": "pain", "rash": "rash",
        "diabetes": "diabetes", "hypertension": "high blood pressure",
        "asthma": "asthma", "arthritis": "arthritis"
    }
    
    for condition, topic in medical_conditions.items():
        if condition in question_lower:
            return [
                f"What are the symptoms of {topic}?",
                f"How is {topic} treated?",
                f"What causes {topic}?"
            ]
    
    return []

def generate_prompt(intent, context, question):
    # Clean the question for topic extraction
    cleaned_question = re.sub(r'\b(i am|i have|my|me|father|mother|child|husband|wife)\b', '', question.lower(), flags=re.IGNORECASE)
    
    # Extract symptoms and medical terms
    symptoms = re.findall(r'\b(headache|fever|pain|nausea|vomiting|dizziness|fatigue|cough|sore throat|rash|swelling)\b', cleaned_question, re.IGNORECASE)
    medical_terms = re.findall(r'\b([a-zA-Z]{4,})\b', cleaned_question)
    
    if symptoms:
        topic = " and ".join(symptoms[:2]) + " symptoms"
    elif medical_terms:
        topic = medical_terms[0] if len(medical_terms[0]) > 4 else "this condition"
    else:
        topic = "your symptoms"

    if intent == "personal_case":
        # Extract age and symptoms for personal cases
        age_match = re.search(r'(\d+)\s*(year|yr)s?\s*old', question.lower())
        age_info = f"**Age:** {age_match.group(1)} years\n" if age_match else ""
        
        symptom_list = re.findall(r'\b(headache|fever|pain|nausea|vomiting|dizziness|fatigue|cough|sore throat|rash|swelling)\b', question.lower(), re.IGNORECASE)
        symptoms_str = ", ".join(symptom_list) if symptom_list else "symptoms"
        
        return f"""Provide a concise medical assessment based on:

Patient Presentation:
{age_info}**Symptoms:** {symptoms_str}

Context: {context}

Required Structure:
👨‍⚕️ **CLINICAL ASSESSMENT**
• Brief symptom analysis
• 2-3 most likely causes

⚕️ **RECOMMENDED ACTIONS**
• Immediate self-care measures
• When to seek medical help
• What to tell your doctor

📋 **KEY RECOMMENDATIONS**
• Most important advice

CRITICAL RULES:
1. **DO NOT invent medical history** - only use what's stated
2. **DO NOT include statistics** - focus on practical advice
3. **Keep response under 200 words**
4. **Be specific to the described symptoms only**
5. **Avoid listing unrelated conditions**
6. **Use simple, clear language**
7. **Focus on actionable advice**
8. **Never prescribe specific medications**
9. **Always recommend professional consultation**
10. **Keep it concise and relevant**"""

    elif intent == "report_analysis":
        return f"""Analyze this medical report:

Report Content: {context}
Patient Question: {question}

Required Structure:
📄 **REPORT ANALYSIS**
• Key findings summary
• What the results mean

🎯 **RECOMMENDED NEXT STEPS**
• Follow-up actions
• When to consult specialist

💡 **PATIENT GUIDANCE**
• How to interpret results
• Questions for healthcare provider

CRITICAL RULES:
1. **Be accurate and conservative**
2. **Do not over-interpret results**
3. **Recommend professional consultation**
4. **Keep response under 250 words**"""

    else:
        return f"""Provide a professional medical response:

Question: {question}
Context: {context}

Required Structure:
🩺 **INFORMATION**
• Clear, concise explanation
• Key facts

💡 **PRACTICAL ADVICE**
• Actionable recommendations
• When to seek help

📋 **KEY POINTS**
• Most important information

CRITICAL RULES:
1. **Be accurate and evidence-based**
2. **Keep response under 200 words**
3. **Use simple language**
4. **Focus on practical advice**
5. **Recommend professional consultation when appropriate**
6. **Do not include statistics unless essential**
7. **Avoid technical jargon unless explained**"""

def load_faiss_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.exists("vectorstore_faiss"):
        return FAISS.from_texts(["Initial empty document"], embeddings)
    return FAISS.load_local("vectorstore_faiss", embeddings, allow_dangerous_deserialization=True)

def extract_pdf_text(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = "\n".join([p.extract_text() or '' for p in reader.pages])
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
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
                    pdf_results = vector_store.similarity_search(user_query, k=2)
                    pdf_context = "\n\n".join([r.page_content for r in pdf_results])
                else:
                    pdf_context = ""
                
                general_results = vector_store.similarity_search(user_query, k=2)
                general_context = "\n\n".join([r.page_content for r in general_results])
                
                if pdf_context:
                    context = f"PATIENT RECORD:\n{pdf_context}\n\nMEDICAL KNOWLEDGE:\n{general_context}"
                else:
                    context = general_context
                
                # Clean context to prevent hallucinations
                context = context[:800]
                context = re.sub(r'\.\s*\d+\.\s*\d+\.\s*\d+', '', context)
                
                fallback = not context.strip() or len(context.strip()) < 30
                
                try:
                    groq_chat = ChatOpenAI(
                        openai_api_key=st.secrets.get("GROQ_API_KEY", ""),
                        base_url="https://api.groq.com/openai/v1",
                        model_name="llama-3.3-70b-versatile",
                        temperature=0.1,  # Lower temperature for less creativity
                        request_timeout=30
                    )
                    
                    if fallback:
                        try:
                            tavily = TavilySearchResults(api_key=st.secrets.get("TAVILY_API_KEY", ""))
                            trusted_domains = get_trusted_domains(user_query)
                            search_results = tavily.run({
                                "query": user_query,
                                "include_domains": trusted_domains,
                                "num_results": 3
                            })
                            context = "\n\n".join([r["content"] for r in search_results])
                        except:
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