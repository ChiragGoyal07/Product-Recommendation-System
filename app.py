import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from rag_system import RAGSystem
from PIL import Image


# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Skincare Product Recommender",
    page_icon="💫",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    /* Base app styling */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Chat message styling - modernized */
    .chat-message {
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1.5rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        gap: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    
    .chat-message:hover {
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.08);
        transform: translateY(-2px);
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #212529;
        border-left: 4px solid #4287f5;
    }
    
    .chat-message.assistant {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        color: #0d47a1;
        border-left: 4px solid #2962ff;
    }
    
    .chat-message .avatar {
        min-width: 3rem;
        height: 3rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        background: white;
        border-radius: 50%;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .chat-message .content {
        width: 100%;
        padding: 0;
        color: #333333;
        line-height: 1.6;
    }
    
    /* Custom header styling - modernized */
    .main-header {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 3rem 2.5rem;
        border-radius: 1.5rem;
        margin: 2rem 0 3rem 0;
        text-align: center;
        border: none;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.3) 100%);
        pointer-events: none;
    }
    
    .main-header h1 {
        font-weight: 800;
        margin-bottom: 1.5rem;
        color: #1a365d;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        position: relative;
    }
    
    .main-header p {
        font-size: 1.2rem;
        color: #2d3748;
        max-width: 700px;
        margin: 0 auto;
        line-height: 1.6;
        position: relative;
        font-weight: 500;
    }

    .main-header .sparkle {
        position: absolute;
        width: 20px;
        height: 20px;
        background: radial-gradient(circle, #fff 0%, rgba(255,255,255,0) 70%);
        border-radius: 50%;
        animation: sparkle 2s infinite;
    }

    @keyframes sparkle {
        0% { transform: scale(0); opacity: 0; }
        50% { transform: scale(1); opacity: 0.5; }
        100% { transform: scale(0); opacity: 0; }
    }
    
    /* Lists and content inside messages */
    .chat-message ul, .chat-message ol {
        padding-left: 1.5rem;
        color: #333333;
    }
    
    .chat-message li {
        margin-bottom: 0.75rem;
    }
    
    .chat-message h3 {
        color: #1a365d;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }
    
    /* Enhance link styling */
    .chat-message a {
        color: #3182ce;
        text-decoration: none;
        border-bottom: 1px dotted #3182ce;
        transition: all 0.2s ease;
    }
    
    .chat-message a:hover {
        color: #2c5282;
        border-bottom: 1px solid #2c5282;
    }
    
    /* Product cards */
    .product-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        color: #333333;
        border-left: 3px solid #3182ce;
        transition: all 0.2s ease;
    }
    
    .product-card:hover {
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.08);
        transform: translateY(-2px);
    }
    
    /* Streamlit elements customization */
    .stButton button {
        border-radius: 0.5rem;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }
    
    .stTextInput input {
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        border: 1px solid #e2e8f0;
        font-size: 1rem;
    }
    
    .stTextInput input:focus {
        border-color: #3182ce;
        box-shadow: 0 0 0 1px #3182ce;
    }
    
    /* Force content coloring */
    .chat-message * {
        color: inherit;
    }
    
    .chat-message.user * {
        color: #212529;
    }
    
    .chat-message.assistant * {
        color: #0d47a1;
    }
    
    .chat-message.assistant li, 
    .chat-message.assistant p {
        color: #2d3748;
    }
    
    .chat-message.assistant h3 {
        color: #1a365d;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to display chat messages
def display_message(message, is_user=False):
    if is_user:
        avatar = "👤"
        style = "user"
    else:
        avatar = "🤖"
        style = "assistant"
    
    st.markdown(f"""
    <div class="chat-message {style}">
        <div class="avatar">
            {avatar}
        </div>
        <div class="content">
            {message}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Initialize session state variables for chat
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "rag_system" not in st.session_state:
    # Get API keys from environment variables
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "tavily_api_key": os.getenv("TAVILY_API_KEY"),
        "weaviate_cluster_url": os.getenv("WEAVIATE_CLUSTER_URL"),
        "weaviate_api_key": os.getenv("WEAVIATE_API_KEY"),
        "product_data_path": os.getenv("PRODUCT_DATA_PATH")
    }
    
    st.session_state.rag_system = RAGSystem(config)
    st.session_state.system_initialized = False
    st.session_state.data_processed = False
    print("RAG system initialized successfully")

# Define a callback to clear the input after sending
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

def clear_input_callback():
    st.session_state.clear_input = True

# Sidebar for system setup
with st.sidebar:
    st.title("System Setup")
    
    if not st.session_state.system_initialized:
        if st.button("Initialize System"):
            with st.spinner("Initializing RAG system..."):
                st.session_state.rag_system.initialize()
                st.session_state.system_initialized = True
            st.success("System initialized successfully!")
    else:
        st.success("System initialized ✅")
    
    if st.session_state.system_initialized and not st.session_state.data_processed:
        if st.button("Process Product Data"):
            with st.spinner("Processing product data and loading into vector store..."):
                st.session_state.rag_system.process_data()
                st.session_state.data_processed = True
            st.success("Data processed and loaded successfully!")
    elif st.session_state.data_processed:
        st.success("Data processed ✅")
    
    if st.session_state.data_processed:
        if st.button("Reset Conversation"):
            st.session_state.rag_system.reset_conversation()
            st.session_state.messages = []
            st.experimental_rerun()
    
    st.divider()
    st.markdown("""
    ### About this System
    
    This chatbot uses Retrieval Augmented Generation (RAG) to provide:
    
    1. **Product Recommendations** - Ask for skincare products suited to your needs
    2. **General Skincare Advice** - Get answers to skincare questions and concerns
    
    The system automatically detects your query type and routes it to the appropriate source.
    """)

# Load and display logo
logo_path = "clinikally_cover.jpeg"
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.image(logo, use_container_width=True)
# Main header
st.markdown("""
<div class="main-header">
    <div class="sparkle" style="top: 20%; left: 20%;"></div>
    <div class="sparkle" style="top: 60%; right: 25%;"></div>
    <div class="sparkle" style="bottom: 30%; left: 40%;"></div>
    <h1>🧴Derma Product Recommender</h1>
    <p>Ask for personalized skincare product recommendations or general skincare advice!</p>
</div>
""", unsafe_allow_html=True)

# Check if system is ready for queries
system_ready = st.session_state.system_initialized and st.session_state.data_processed

if not system_ready:
    st.warning("⚠️ Please initialize the system and process the product data using the sidebar controls before asking questions.")
else:
    # Chat interface
    st.subheader("Ask about Skincare")

    # Chat container
    chat_container = st.container()

    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            display_message(message["content"], message["role"] == "user")
        
        # Check if we should clear the input (this happens before the widget is rendered)
        if st.session_state.clear_input:
            # Reset the flag
            st.session_state.clear_input = False
            # This will be used to set the default value of the text input
            query_default = ""
        else:
            # Keep any existing value
            query_default = st.session_state.get("query_input", "")
        
        # Chat input
        query = st.text_input(
            "Ask about skincare products or general skincare advice:", 
            key="query_input", 
            placeholder="e.g., 'Recommend a moisturizer for oily skin under 1000' or 'How to treat acne?'",
            value=query_default
        )
        
        # Process query when submitted
        if query:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Display user message
            display_message(query, is_user=True)
            
            # Process query and get response
            with st.spinner("Thinking..."):
                response = st.session_state.rag_system.process_query(query)
            
            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Display assistant message
            display_message(response)
            
            # Set the clear input flag and rerun to clear the input
            clear_input_callback()
            st.rerun()
            
            # The commented out line below was causing the error
            # st.session_state.query_input = ""  # This line causes the error

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8rem;">
    Powered by OpenAI, Weaviate, and Tavily | Created by Chirag
</div>
""", unsafe_allow_html=True) 