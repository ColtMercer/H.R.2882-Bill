import streamlit as st
from document_processor import DocumentProcessor
from chat_handler import ChatHandler
import os
from dotenv import load_dotenv
import shutil

# Load environment variables
load_dotenv()

# Initialize processors
@st.cache_resource(show_spinner=False)
def init_processors():
    return DocumentProcessor(), ChatHandler()

def cleanup_chroma():
    """Safely cleanup Chroma database"""
    try:
        # Simply remove the directory if it exists
        if os.path.exists('data/chroma_db'):
            shutil.rmtree('data/chroma_db', ignore_errors=True)
    except Exception as e:
        st.error(f"Error cleaning up Chroma: {str(e)}")

def main():
    st.title("Congressional Bill Analysis Chat")
    
    # Initialize processors if not in session state
    if 'doc_processor' not in st.session_state:
        doc_processor, chat_handler = init_processors()
        st.session_state['doc_processor'] = doc_processor
        st.session_state['chat_handler'] = chat_handler
    
    doc_processor = st.session_state['doc_processor']
    chat_handler = st.session_state['chat_handler']
    
    # Initialize vector store if needed
    if not doc_processor.vector_store and os.path.exists('data/bill.txt'):
        with st.spinner("Initializing vector store..."):
            with open('data/bill.txt', 'r', encoding='utf-8') as f:
                bill_text = f.read()
            doc_processor.process_for_vectors(bill_text)

    # Debug info
    st.sidebar.header("Debug Info")
    pdf_path = "data/cr_2_bill.pdf"
    st.sidebar.text(f"PDF Path: {pdf_path}")
    st.sidebar.text(f"PDF exists: {os.path.exists(pdf_path)}")
    
    if os.path.exists('data/bill.txt'):
        with open('data/bill.txt', 'r', encoding='utf-8') as f:
            bill_preview = f.read()[:500] + "..."
        st.sidebar.text("Current Bill Preview:")
        st.sidebar.text(bill_preview)
    
    # Sidebar for full text summary
    with st.sidebar:
        st.header("Full Bill Summary")
        if st.button("Generate Full Summary"):
            with st.spinner("Generating full bill summary..."):
                try:
                    with open('data/bill.txt', 'r', encoding='utf-8') as f:
                        bill_text = f.read()
                    summary = chat_handler.summarize_full_text(bill_text)
                    st.session_state['full_summary'] = summary
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")

        if 'full_summary' in st.session_state:
            st.write(st.session_state['full_summary'])

    # Main chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about the congressional bill"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get relevant context from vector store
                relevant_chunks = doc_processor.get_relevant_chunks(prompt)
                
                # Generate response
                response = chat_handler.answer_question(prompt, relevant_chunks)
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    # Only clear data if it doesn't exist
    if not os.path.exists('data/bill.txt') or not os.path.exists('data/chroma_db'):
        # Clear existing data
        cleanup_chroma()
        
        # Initialize by loading the PDF
        with st.spinner("Loading bill text and initializing vector store..."):
            pdf_path = "data/cr_2_bill.pdf"
            
            if not os.path.exists(pdf_path):
                st.error(f"Could not find PDF file at {pdf_path}. Please verify the file location.")
                st.stop()
            
            st.info(f"Loading PDF from: {pdf_path}")
            doc_processor, chat_handler = init_processors()
            bill_text = doc_processor.load_pdf_text(pdf_path)
            doc_processor.process_for_vectors(bill_text)
            
            # Store in session state
            st.session_state['doc_processor'] = doc_processor
            st.session_state['chat_handler'] = chat_handler
    
    main() 