from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import httpx
import xml.etree.ElementTree as ET
import uvicorn
from groq import Groq  # pip install groq
import os


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load Models & Data
print("Loading models...")
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

sbert_embeddings = np.load('sbert_embeddings.npy')
df = pd.read_csv('papers_metadata.csv')
sbert = SentenceTransformer('all-MiniLM-L6-v2')


groq_client  = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

print("Models and Groq client loaded successfully!")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

def recommend_hybrid(query, top_k=5, alpha=0.3):
    query_vec = tfidf.transform([query])
    tfidf_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    query_emb = sbert.encode([query])
    sbert_scores = cosine_similarity(query_emb, sbert_embeddings).flatten()

    tfidf_max = tfidf_scores.max()
    sbert_max = sbert_scores.max()
    
    tfidf_norm = tfidf_scores / tfidf_max if tfidf_max > 0 else tfidf_scores
    sbert_norm = sbert_scores / sbert_max if sbert_max > 0 else sbert_scores

    hybrid_scores = alpha * tfidf_norm + (1 - alpha) * sbert_norm

    top_indices = hybrid_scores.argsort()[-top_k:][::-1]
    results = df.iloc[top_indices][['title', 'abstract', 'year', 'n_citation']].copy()
    results['score'] = hybrid_scores[top_indices]
    return results

# Live ArXiv Fetcher
async def fetch_arxiv(query: str, top_k: int = 5):
    try:
        clean_query = "+".join(query.split())
        url = f"https://export.arxiv.org/api/query?search_query=all:{clean_query}&max_results={top_k}&sortBy=submittedDate&sortOrder=descending"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=5)
            
        if response.status_code != 200:
            return []
            
        root = ET.fromstring(response.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        for entry in root.findall('atom:entry', ns):
            title_node = entry.find('atom:title', ns)
            summary_node = entry.find('atom:summary', ns)
            id_node = entry.find('atom:id', ns)
            pub_node = entry.find('atom:published', ns)
            
            papers.append({
                "title": title_node.text.strip().replace('\n', ' ') if title_node is not None else "No Title",
                "abstract": summary_node.text.strip().replace('\n', ' ') if summary_node is not None else "No Abstract",
                "link": id_node.text.strip() if id_node is not None else "#",
                "published": pub_node.text[:10] if pub_node is not None else "",
                "source": "ArXiv"
            })
        return papers
    except Exception as e:
        print(f"ArXiv Fetch Error: {e}")
        return []

# 3. Endpoints
@app.get("/")
def root():
    return {"message": "Research Paper Recommender API is running Live! 🚀"}

@app.post("/recommend")
async def recommend(request: QueryRequest):
    try:
        hybrid_results = recommend_hybrid(request.query, request.top_k)
        hybrid_list = []
        for _, row in hybrid_results.iterrows():
            hybrid_list.append({
                "title": row["title"],
                "abstract": row["abstract"],
                "year": int(row["year"]) if pd.notna(row["year"]) else None,
                "n_citation": int(row["n_citation"]) if pd.notna(row["n_citation"]) else 0,
                "score": round(float(row["score"]), 4),
                "source": "Local DB"
            })
    except Exception as e:
        print(f"Local Search Error: {e}")
        hybrid_list = []

    arxiv_list = await fetch_arxiv(request.query, request.top_k)

    return {
        "query": request.query,
        "local_results": hybrid_list,
        "arxiv_results": arxiv_list
    }

@app.post("/summarize")
async def summarize_paper(abstract: str):
    try:
        system_instruction = (
            "You are an expert scientific and medical research assistant. "
            "Summarize the following abstract into 3 clear, concise bullet points (Key Takeaways) in English. "
            "Maintain important domain terms but ensure clarity."
        )
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": abstract}
            ],
            model="llama-3.1-8b-instant", 
            temperature=0.2, 
            max_tokens=300 
        )
        
        summary = chat_completion.choices[0].message.content
        return {"summary": summary}
    except Exception as e:
        print(f"Groq Error Details: {e}")
        return {"summary": f"Llama 3 Error: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)