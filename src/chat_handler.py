from openai import OpenAI
from typing import List, Dict

class ChatHandler:
    def __init__(self):
        self.client = OpenAI()
        self.max_tokens = 30000  # Conservative limit for GPT-4

    def chunk_text(self, text: str, max_tokens: int = 30000) -> List[str]:
        """Split text into chunks that won't exceed token limit"""
        # Rough approximation: 1 token = 4 chars
        chars_per_chunk = max_tokens * 4
        
        # Split into paragraphs first
        paragraphs = text.split('\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < chars_per_chunk:
                current_chunk += para + "\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para + "\n"
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def summarize_full_text(self, text: str) -> str:
        """Summarize the entire bill text in chunks"""
        try:
            # Split text into manageable chunks
            chunks = self.chunk_text(text)
            
            # Get summary for each chunk
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes congressional bills clearly and concisely."},
                        {"role": "user", "content": f"Please summarize this section (Part {i+1} of {len(chunks)}) of the bill: {chunk}"}
                    ],
                    temperature=0.3
                )
                chunk_summaries.append(response.choices[0].message.content)
            
            # Combine chunk summaries into final summary
            if len(chunk_summaries) > 1:
                combined_summary = "\n\n".join(chunk_summaries)
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that combines section summaries into a coherent overall summary."},
                        {"role": "user", "content": f"Please combine these section summaries into one coherent summary:\n\n{combined_summary}"}
                    ],
                    temperature=0.3
                )
                return response.choices[0].message.content
            else:
                return chunk_summaries[0]
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def answer_question(self, question: str, context: List[str]) -> str:
        """Answer questions based on relevant context"""
        context_text = "\n".join(context)
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions about congressional bills based on the provided context."},
                    {"role": "user", "content": f"Context: {context_text}\n\nQuestion: {question}"}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error answering question: {str(e)}" 