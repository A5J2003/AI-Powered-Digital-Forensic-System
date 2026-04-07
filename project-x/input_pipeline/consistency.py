from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def compute_similarity(text1, text2):
    text1 = clean_text(text1)
    text2 = clean_text(text2)

    emb1 = semantic_model.encode([text1])
    emb2 = semantic_model.encode([text2])

    score = cosine_similarity(emb1, emb2)[0][0]
    return float(score)