import streamlit as st
import requests

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Research Assistant",
    page_icon="🧠",
    layout="wide"
)

# CSS for classy UI.
st.markdown("""
<style>
    /* Chat bubbles */
    .stChatMessage {
        border-radius: 10px;
    }
    
    /* Header */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #2e3b4e;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar UI
with st.sidebar:
    st.title("⚙️ Knowledge Base Setup")
    
    st.markdown("### 1. Document Upload")
    uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
    
    if st.button("📤 Upload PDFs", use_container_width=True):
        if uploaded_files:
            with st.spinner("Uploading files..."):
                success_count = 0
                for file in uploaded_files:
                    files = {"file": (file.name, file.getvalue(), "application/pdf")}
                    try:
                        response = requests.post(f"{API_BASE_URL}/upload", files=files)
                        if response.status_code == 200:
                            success_count += 1
                    except Exception as e:
                        st.error(f"Failed to upload {file.name}: {e}")
                
                if success_count > 0:
                    st.success(f"Successfully uploaded {success_count} files!")
        else:
            st.warning("Please select files to upload first.")
            
    st.markdown("### 2. Processing Strategy")
    parsing_strategy = st.select_slider(
        "Select Parsing Strategy",
        options=["fast", "medium", "deep"],
        value="medium",
        help="Fast: Basic text extraction. Medium: Balances speed and quality. Deep: Uses vision models for charts and complex layouts."
    )
    
    if st.button("🧠 Create Knowledge Base", type="primary", use_container_width=True):
        with st.spinner("Initializing Knowledge Base processing..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/Knowledge_base", 
                    params={"parsing_strategy": parsing_strategy}
                )
                if response.status_code == 200:
                    st.success(response.json().get("message", "Knowledge base creation started!"))
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
                
    st.divider()
    
    st.markdown("### 🔄 Session Management")
    if st.button("💬 Start New Chat", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()

# Main Chat UI
st.title("🧠 Research ReAct Agent")
st.markdown("Ask complex research questions, generate reports, and interact with your uploaded knowledge base.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "report" in message and message["report"]:
            with st.expander("📄 View Generated Research Report"):
                st.markdown(message["report"])

# Chat Input
if prompt := st.chat_input("Ask a research question..."):
    # Display user prompt
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Append to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Call Backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking and researching..."):
            try:
                payload = {
                    "query": prompt,
                    "session_id": st.session_state.session_id
                }
                
                response = requests.post(f"{API_BASE_URL}/research_chat", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Update session ID if it's a new session
                    if data.get("is_new_session"):
                        st.session_state.session_id = data.get("session_id")
                        
                    ai_response = data.get("response", "No response provided.")
                    research_report = data.get("research_report", "")
                    
                    st.markdown(ai_response)
                    if research_report:
                        with st.expander("📄 View Generated Research Report"):
                            st.markdown(research_report)
                            
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": ai_response,
                        "report": research_report
                    })
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend server. Please make sure the FastAPI server is running on http://localhost:8000")
            except Exception as e:
                st.error(f"An error occurred: {e}")
