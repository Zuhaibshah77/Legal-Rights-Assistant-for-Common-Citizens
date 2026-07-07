from flask import Flask, render_template, request, jsonify
import pickle
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

app = Flask(__name__)

# ── Load everything once at startup ───────────────────────
print("Loading system...")

with open('models/classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)
with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

embedder   = SentenceTransformer('all-MiniLM-L6-v2')
client     = chromadb.PersistentClient(path="vectorstore")
collection = client.get_collection("legal_rights")

from dotenv import load_dotenv
import os
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client  = Groq(api_key=GROQ_API_KEY)

print("✅ System ready!")

def get_legal_answer(user_query):
    # ML Classification
    query_vec  = vectorizer.transform([user_query])
    domain     = classifier.predict(query_vec)[0]
    proba      = classifier.predict_proba(query_vec)[0]
    confidence = round(max(proba) * 100, 1)

    # RAG Retrieval
    query_embedding = embedder.encode([user_query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    context = ""
    sources = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        source = meta.get('source', '').split('/')[-1].replace('.pdf', '').replace('_', ' ').title()
        context += f"\n\nSource ({source}):\n{doc}"
        sources.append(source)

    # Groq LLM
    prompt = f"""You are a Legal Rights Assistant helping common Indian citizens understand their legal rights in simple language.

A citizen has this problem: "{user_query}"

Legal domain identified: {domain}

Relevant sections from Indian law:
{context}

Please provide:
1. A simple explanation of their legal rights in this situation
2. Specific steps they can take (numbered list)
3. Which law or section protects them
4. Whether they need a lawyer or can handle it themselves

Use simple, clear language that any common citizen can understand.
End with: "⚠️ Disclaimer: This is general legal information, not professional legal advice."
"""

    response = groq_client.chat.completions.create(
        model      = "llama-3.3-70b-versatile",
        messages   = [{"role": "user", "content": prompt}],
        max_tokens = 1024
    )

    answer = response.choices[0].message.content

    return {
        "answer"    : answer,
        "domain"    : domain,
        "confidence": confidence,
        "sources"   : list(set(sources))
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data       = request.get_json()
    user_query = data.get('query', '')
    
    if not user_query.strip():
        return jsonify({"error": "Please enter a question"})
    
    result = get_legal_answer(user_query)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)