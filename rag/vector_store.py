# rag/vector_store.py

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import os

class VectorStore:
    def __init__(
        self,
        collection_name: str,
        embedding_model: str,
        persist_directory: str = "./data/chromadb"
    ):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(collection_name)
            print(f"✅ Loaded existing collection: {collection_name}")
        except ValueError:
            self.collection = self.client.create_collection(collection_name)
            print(f"✅ Created new collection: {collection_name}")

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """Add documents to the vector store."""
        ids = [f"doc_{i}" for i in range(len(texts))]
        embeddings = self.embedding_model.encode(texts).tolist()
        
        if metadatas is None:
            metadatas = [{"chunk_id": i} for i in range(len(texts))]
        
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"📥 Added {len(texts)} documents to collection.")

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve the top-k most similar documents."""
        query_embedding = self.embedding_model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances", "metadatas"]
        )
        
        # Format results
        docs = []
        for i, doc in enumerate(results['documents'][0]):
            distance = results['distances'][0][i]
            if score_threshold is not None and distance > score_threshold:
                continue
            docs.append({
                'text': doc,
                'distance': distance,
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
            })
        return docs

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieve all documents from the collection for building BM25."""
        # Get the total count
        total = self.collection.count()
        if total == 0:
            return []
        
        # Query all documents (ChromaDB limits to 10 by default, so we specify n_results)
        results = self.collection.get(
            include=["documents", "metadatas"]
        )
        
        docs = []
        for i, doc in enumerate(results['documents']):
            docs.append({
                'text': doc,
                'metadata': results['metadatas'][i] if results['metadatas'] else {}
            })
        return docs