from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import PyPDF2
import os
from typing import List, Dict

class DocumentProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vector_store = None

    def __del__(self):
        """Cleanup when the object is destroyed"""
        if self.vector_store:
            try:
                self.vector_store._client.close()
            except:
                pass

    def load_pdf_text(self, file_path: str) -> str:
        """Load and parse text from a PDF file"""
        with open(file_path, 'rb') as file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from all pages
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text() + '\n'
            
            # Clean up the text
            text = ' '.join(line.strip() for line in text.split('\n') if line.strip())
            
            # Save the text
            os.makedirs('data', exist_ok=True)
            with open('data/bill.txt', 'w', encoding='utf-8') as f:
                f.write(text)
            
            return text

    def process_for_vectors(self, text: str):
        """Process the document for vector storage"""
        chunks = self.text_splitter.split_text(text)
        self.vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            persist_directory="./data/chroma_db"
        )
        return self.vector_store

    def get_relevant_chunks(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant chunks for a query"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        results = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results] 