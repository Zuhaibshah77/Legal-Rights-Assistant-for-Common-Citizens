import pickle
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# ── Load ML Model ─────────────────────────────────────────
print("Loading ML classifier...")
with open('models/classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)
with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# ── Load RAG System ───────────────────────────────────────
print("Loading RAG system...")
embedder   = SentenceTransformer('all-MiniLM-L6-v2')
client     = chromadb.PersistentClient(path="vectorstore")
collection = client.get_collection("legal_rights")

# ── Groq API ──────────────────────────────────────────────
from dotenv import load_dotenv
import os
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("✅ System ready!\n")

def query_legal_assistant(user_query):
    print(f"{'='*60}")
    print(f"Query: {user_query}")
    print(f"{'='*60}")

    # ── Step 1: ML Classification ─────────────────────────
    query_vec  = vectorizer.transform([user_query])
    domain     = classifier.predict(query_vec)[0]
    proba      = classifier.predict_proba(query_vec)[0]
    confidence = max(proba) * 100

    print(f"\n[ML Classifier]")
    print(f"  Domain     : {domain}")
    print(f"  Confidence : {confidence:.1f}%")

    # ── Step 2: RAG Retrieval ─────────────────────────────
    query_embedding = embedder.encode([user_query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    context = ""
    print(f"\n[RAG Retrieval]")
    for i, (doc, meta) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0]
    )):
        source = meta.get('source', '').split('/')[-1]
        print(f"  [{i+1}] {source}: {doc[:80]}...")
        context += f"\n\nSource {i+1} ({source}):\n{doc}"

    # ── Step 3: Generate Answer with Groq ─────────────────
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

    print(f"\n[Generating answer with Groq...]")

    response = groq_client.chat.completions.create(
        model    = "llama-3.3-70b-versatile",
        messages = [{"role": "user", "content": prompt}],
        max_tokens = 1024
    )

    answer = response.choices[0].message.content

    print(f"\n{'='*60}")
    print(f"LEGAL ASSISTANT ANSWER:")
    print(f"{'='*60}")
    print(answer)

    return answer, domain, confidence

# ── Test queries ──────────────────────────────────────────
query_legal_assistant("My employer has not paid my salary for 3 months")
print("\n")
query_legal_assistant("My landlord is not returning my security deposit")