from sentence_transformers import SentenceTransformer
import chromadb

# Load embedder and ChromaDB
print("Loading RAG system...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
client     = chromadb.PersistentClient(path="vectorstore")
collection = client.get_collection("legal_rights")

print(f"Total chunks in DB: {collection.count()}")

def retrieve(query, top_k=3):
    # Embed the query
    query_embedding = embedder.encode([query])[0].tolist()
    
    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    print(f"\n🔍 Query: {query}")
    print(f"\n--- Top {top_k} Relevant Chunks ---")
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"\n[{i+1}] Domain: {meta.get('domain', 'Unknown')}")
        print(f"Source: {meta.get('source', 'Unknown')}")
        print(f"Text: {doc[:300]}...")
        print("-" * 50)

# Test queries
retrieve("What are the rights of a worker who is dismissed?")
retrieve("How is income tax calculated on salary?")
retrieve("What is the procedure to file a civil suit?")