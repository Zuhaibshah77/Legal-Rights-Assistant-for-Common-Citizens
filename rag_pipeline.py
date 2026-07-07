import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import pickle

# ── 1. Load all PDFs ──────────────────────────────────────
print("Loading PDFs...")

pdf_files = [
    ('Criminal Law'         , 'data/laws/industrial_disputes_act.pdf'),
    ('Criminal Law'         , 'data/laws/ipc.pdf'),
    ('Tax Law'              , 'data/laws/income_tax_act.pdf'),
    ('Civil Law'            , 'data/laws/civil_procedure_code.pdf'),
    ('Civil Law'            , 'data/laws/transfer_of_property_act.pdf'),
    ('Consumer Law'         , 'data/laws/consumer_protection_act.pdf'),
    ('Constitutional Law'   , 'data/laws/constitution_of_india.pdf'),
    ('RTI Law'              , 'data/laws/rti_act.pdf'),
]
all_documents = []

for domain, path in pdf_files:
    print(f"  Loading {domain} from {path}...")
    loader = PyPDFLoader(path)
    docs   = loader.load()

    for doc in docs:
        doc.metadata['domain'] = domain

    all_documents.extend(docs)
    print(f"  Loaded {len(docs)} pages")

print(f"\nTotal pages loaded: {len(all_documents)}")

# ── 2. Split into chunks ──────────────────────────────────
print("\nSplitting into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size    = 500,
    chunk_overlap = 50,
    separators    = ["\n\n", "\n", ".", " "]
)

chunks = splitter.split_documents(all_documents)
print(f"Total chunks created: {len(chunks)}")

# ── 3. Create Embeddings ──────────────────────────────────
print("\nLoading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded!")

texts     = [chunk.page_content for chunk in chunks]
metadatas = [chunk.metadata for chunk in chunks]

print(f"\nCreating embeddings for {len(texts)} chunks...")
print("This will take several minutes...")
embeddings = embedder.encode(texts, show_progress_bar=True)
print(f"Embeddings shape: {embeddings.shape}")

# ── 4. Store in ChromaDB ──────────────────────────────────
print("\nSetting up ChromaDB...")
client = chromadb.PersistentClient(path="vectorstore")

# Delete old collection and recreate fresh
try:
    client.delete_collection("legal_rights")
    print("Old collection deleted")
except:
    print("No old collection found")

collection = client.get_or_create_collection(
    name     = "legal_rights",
    metadata = {"hnsw:space": "cosine"}
)

print("Storing chunks in ChromaDB...")
batch_size = 100
for i in range(0, len(texts), batch_size):
    batch_texts      = texts[i:i+batch_size]
    batch_embeddings = embeddings[i:i+batch_size].tolist()
    batch_metadatas  = metadatas[i:i+batch_size]
    batch_ids        = [f"chunk_{j}" for j in range(i, i+len(batch_texts))]

    collection.add(
        documents  = batch_texts,
        embeddings = batch_embeddings,
        metadatas  = batch_metadatas,
        ids        = batch_ids
    )
    print(f"  Stored chunks {i} to {i+len(batch_texts)}")

print(f"\n✅ Total chunks stored in ChromaDB: {collection.count()}")

# ── 5. Save embedder name ─────────────────────────────────
with open('models/embedder_name.pkl', 'wb') as f:
    pickle.dump('all-MiniLM-L6-v2', f)

print("✅ RAG pipeline rebuilt successfully!")
print("Vectorstore ready at: vectorstore/")