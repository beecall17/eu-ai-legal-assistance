# scripts/ingest_data.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
import pymupdf4llm
from rag.vector_store import VectorStore
from config.settings import EMBEDDING_MODEL, CHROMA_DB_PATH
from tqdm import tqdm

def load_and_chunk_pdf(
    file_path: str,
    child_chunk_size: int = 400,
    child_chunk_overlap: int = 50
) -> list[Document]:
    """
    Load a PDF, convert to markdown, then split hierarchically.
    
    Returns:
        A list of child Document objects (the fine-grained chunks used for retrieval).
        Each child Document contains:
            - page_content: The text of the chunk.
            - metadata: Includes the parent header hierarchy (Chapter/Article/Annex).
    """
    # 1. Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at {file_path}. Please place eu_ai_act.pdf in data/raw/")

    # 2. Convert PDF to Markdown (preserves headings)
    print(f"📄 Loading PDF from: {file_path}")
    md_text = pymupdf4llm.load_pdf(
        file_path,
        remove_headers_footers=True
    )
    print(f"✅ Loaded PDF. Markdown length: {len(md_text)} characters.")

    # 3. Split by structural headers (Chapters, Articles, Annexes)
    headers_to_split_on = [
        ("#", "CHAPTER"),
        ("##", "Article"),
        ("#", "ANNEX")
    ]
    
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False  # Keep the header text in the content
    )
    
    parent_docs = header_splitter.split_text(md_text)
    print(f"✅ Created {len(parent_docs)} structural parent sections (Chapters/Articles).")

    # 4. Split each parent into smaller child chunks for vector search
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    all_child_docs = []
    for parent in tqdm(parent_docs, desc="Splitting parents into children"):
        # Create child chunks from the parent's content.
        # We pass the parent's metadata to each child so we know which Article it belongs to.
        children = child_splitter.create_documents(
            texts=[parent.page_content],
            metadatas=[parent.metadata]  # Propagate the header hierarchy to every child
        )
        all_child_docs.extend(children)
    
    print(f"✅ Created {len(all_child_docs)} child chunks for vector indexing.")
    
    # Optional: Print a sample to verify
    if all_child_docs:
        sample = all_child_docs[0]
        print(f"\n📌 Sample child chunk metadata: {sample.metadata}")
        print(f"📝 Sample child chunk preview: {sample.page_content[:150]}...\n")
    
    return all_child_docs

def ingest():
    """Main ingestion pipeline: Load PDF -> Chunk -> Embed -> Store in ChromaDB."""
    
    # 1. Load and chunk the PDF
    child_docs = load_and_chunk_pdf("data/raw/eu_ai_act.pdf")
    
    if not child_docs:
        print("❌ No chunks generated. Exiting.")
        return
    
    # 2. Initialize Vector Store (creates the DB if it doesn't exist)
    vs = VectorStore(
        collection_name="eu_ai_act",
        embedding_model=EMBEDDING_MODEL,
        persist_directory=CHROMA_DB_PATH
    )
    
    # 3. Extract texts and metadata for the vector store
    texts = [doc.page_content for doc in child_docs]
    metadatas = [doc.metadata for doc in child_docs]
    
    # 4. Add documents
    vs.add_documents(texts, metadatas=metadatas)
    print(f"✅ Successfully ingested {len(texts)} chunks into ChromaDB at {CHROMA_DB_PATH}")

if __name__ == "__main__":
    ingest()