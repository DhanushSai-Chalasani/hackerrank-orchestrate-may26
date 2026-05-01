import os
import glob
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

class DocumentRetriever:
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        self.documents = [] # List of dicts: {'company': c, 'filepath': f, 'content': text, 'chunk_id': id}
        print("Loading embedding model (this may take a moment)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings_matrix = None
        self._load_and_chunk_documents()

    def _clean_text(self, text):
        # Remove markdown formatting, links, etc.
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text) # Remove links but keep text
        text = re.sub(r'[#*`_]', '', text)
        return text

    def _chunk_text(self, text, filepath, company, chunk_size=1000, overlap=200):
        # Simple character-based chunking with overlap
        chunks = []
        start = 0
        text = self._clean_text(text)
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append({
                'company': company.lower(),
                'filepath': filepath,
                'content': chunk
            })
            start += chunk_size - overlap
        return chunks

    def _load_and_chunk_documents(self):
        print(f"Loading and chunking documents from {self.data_dir}...")
        companies = ['hackerrank', 'claude', 'visa']
        
        for company in companies:
            path_pattern = os.path.join(self.data_dir, company, '**', '*.md')
            # glob recursion requires python 3.5+
            files = glob.glob(path_pattern, recursive=True)
            for filepath in files:
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                        chunks = self._chunk_text(text, filepath, company)
                        self.documents.extend(chunks)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
                    
        print(f"Total chunks created: {len(self.documents)}")
        
        # Build Semantic Index
        texts = [doc['content'] for doc in self.documents]
        print("Encoding documents...")
        self.embeddings_matrix = self.model.encode(texts, show_progress_bar=False)
        print("Semantic Index built successfully.")

    def retrieve(self, query, company=None, top_k=5):
        query_vec = self.model.encode([self._clean_text(query)])
        
        # Calculate cosine similarity against all chunks
        similarities = cosine_similarity(query_vec, self.embeddings_matrix).flatten()
        
        # Get top indices sorted by highest similarity
        top_indices = similarities.argsort()[::-1]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            
            # Filter by company if specified
            if company and company.lower() != 'none':
                if doc['company'] != company.lower():
                    continue
                    
            results.append({
                'content': doc['content'],
                'score': similarities[idx],
                'company': doc['company'],
                'filepath': doc['filepath']
            })
            
            if len(results) >= top_k:
                break
                
        return results
