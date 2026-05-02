import os
import glob
import hashlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re

# Tell HuggingFace libraries to never make network calls — use only local cache.
# Fixes: '[Errno 11001] getaddrinfo failed' / 'Cannot send a request, as the
# client has been closed' errors when the model is already cached locally.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Embeddings cache: the first run encodes 5000+ chunks (~3-5 min on CPU).
# Subsequent runs load the pre-computed matrix from disk in < 1 second.
CACHE_DIR  = os.path.join(os.path.dirname(__file__), ".embeddings_cache")
MODEL_NAME = "all-MiniLM-L6-v2"


class DocumentRetriever:
    def __init__(self, data_dir="../data"):
        self.data_dir   = data_dir
        self.documents  = []  # list of dicts: {company, filepath, content}

        print("Loading embedding model (offline cache)...")
        # local_files_only=True: skip the HuggingFace version-check HTTP call
        self.model = SentenceTransformer(MODEL_NAME, local_files_only=True)

        self.embeddings_matrix = None
        self._load_and_chunk_documents()

    # ─── Text helpers ────────────────────────────────────────────────────────

    def _clean_text(self, text):
        """Strip markdown formatting while preserving readable content."""
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # links → text
        text = re.sub(r'[#*`_]', '', text)
        return text.strip()

    def _chunk_text(self, text, filepath, company, chunk_size=1000, overlap=200):
        """Character-based chunking with overlap."""
        chunks = []
        text   = self._clean_text(text)
        start  = 0
        while start < len(text):
            chunks.append({
                'company':  company.lower(),
                'filepath': filepath,
                'content':  text[start:start + chunk_size],
            })
            start += chunk_size - overlap
        return chunks

    # ─── Document loading ────────────────────────────────────────────────────

    def _corpus_fingerprint(self):
        """
        Hash the file list + sizes of the corpus so we can detect when the
        corpus changes and invalidate the embedding cache automatically.
        """
        hasher = hashlib.md5()
        for company in ['hackerrank', 'claude', 'visa']:
            pattern = os.path.join(self.data_dir, company, '**', '*.md')
            for fp in sorted(glob.glob(pattern, recursive=True)):
                try:
                    hasher.update(fp.encode())
                    hasher.update(str(os.path.getsize(fp)).encode())
                except Exception:
                    pass
        return hasher.hexdigest()

    def _load_and_chunk_documents(self):
        print(f"Loading and chunking documents from {self.data_dir}...")
        for company in ['hackerrank', 'claude', 'visa']:
            pattern = os.path.join(self.data_dir, company, '**', '*.md')
            for filepath in glob.glob(pattern, recursive=True):
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        self.documents.extend(
                            self._chunk_text(f.read(), filepath, company)
                        )
                except Exception as e:
                    print(f"  [Warning] Could not read {filepath}: {e}")

        total = len(self.documents)
        print(f"Total chunks created: {total}")
        self._build_index(total)

    # ─── Embedding index (with disk cache) ───────────────────────────────────

    def _build_index(self, total):
        """
        Encode all chunks into dense embeddings.
        Uses a disk cache keyed on corpus fingerprint so encoding only runs
        once — subsequent runs load the matrix in < 1 second.
        """
        os.makedirs(CACHE_DIR, exist_ok=True)
        fingerprint = self._corpus_fingerprint()
        cache_file  = os.path.join(CACHE_DIR, f"{fingerprint}.npy")

        if os.path.exists(cache_file):
            print(f"Loading embeddings from cache ({total} chunks)...")
            self.embeddings_matrix = np.load(cache_file)
            print("Semantic index loaded from cache instantly.")
            return

        print(f"Encoding {total} chunks (first run only — will be cached)...")
        texts = [doc['content'] for doc in self.documents]

        # Encode in batches so you can see progress and avoid OOM on large corpora
        batch_size = 256
        batches    = [texts[i:i + batch_size] for i in range(0, total, batch_size)]
        embeddings = []
        for idx, batch in enumerate(batches):
            pct = int((idx + 1) / len(batches) * 100)
            print(f"  Encoding batch {idx+1}/{len(batches)} ({pct}%)...", flush=True)
            embeddings.append(self.model.encode(batch, show_progress_bar=False))

        self.embeddings_matrix = np.vstack(embeddings)
        np.save(cache_file, self.embeddings_matrix)
        print(f"Semantic index built and cached to: {cache_file}")

    # ─── Retrieval ───────────────────────────────────────────────────────────

    def retrieve(self, query, company=None, top_k=5):
        query_vec    = self.model.encode([self._clean_text(query)])
        similarities = cosine_similarity(query_vec, self.embeddings_matrix).flatten()
        top_indices  = similarities.argsort()[::-1]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            # Filter by company unless ticket company is unknown ('none' / 'nan')
            if company and company.lower() not in ('none', 'nan'):
                if doc['company'] != company.lower():
                    continue
            results.append({
                'content':  doc['content'],
                'score':    float(similarities[idx]),
                'company':  doc['company'],
                'filepath': doc['filepath'],
            })
            if len(results) >= top_k:
                break

        return results
