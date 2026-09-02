# scripts/inspect_chunks.py

import sys
import re
from collections import Counter

sys.path.append(".")

from rag.vector_store import VectorStore
from config.settings import EMBEDDING_MODEL, CHROMA_DB_PATH


COLLECTION_NAME = "eu_ai_act"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REQUIRED_METADATA = [
    "chunk_id",
    "section_id",
    "section_type",
    "document",
    "legal_version",
    "page_start",
    "page_end",
]

# Chunks below this character count are suspicious.
MIN_CHUNK_CHARS = 200

# Chunks below this are very likely too small for useful retrieval.
VERY_SMALL_CHUNK_CHARS = 100

# Retrieval questions designed to test different parts of the EU AI Act.
RETRIEVAL_TESTS = [
    {
        "question": "What does Article 5 say about prohibited practices?",
        "expected": ["article_5", "article5", "art_5", "art5"],
    },
    {
        "question": "What are the obligations for providers of high-risk AI systems?",
        "expected": ["article_16", "article_17", "high-risk"],
    },
    {
        "question": "What is the definition of an AI system under the AI Act?",
        "expected": ["article_3"],
    },
    {
        "question": "What transparency obligations apply to AI systems that interact with people?",
        "expected": ["article_50"],
    },
    {
        "question": "What penalties can Member States impose for violations of the AI Act?",
        "expected": ["article_99"],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_all_chunks(vs):
    """
    Fetch all chunks from Chroma.

    Chroma's get() may return a large collection, so this is intended
    primarily for inspection/testing rather than production code.
    """
    results = vs.collection.get(
        include=["documents", "metadatas"]
    )

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    chunks = []

    for i, document in enumerate(documents):
        metadata = metadatas[i] if i < len(metadatas) else {}

        chunks.append({
            "text": document or "",
            "metadata": metadata or {},
        })

    return chunks


def normalize(text):
    """Normalize text for simple comparisons."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def metadata_value(chunk, key, default=""):
    return chunk["metadata"].get(key, default)


def print_separator(title=None):
    print("\n" + "=" * 80)

    if title:
        print(title)
        print("=" * 80)


# ---------------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------------

def test_basic_statistics(chunks):
    print_separator("1. BASIC CHUNK STATISTICS")

    if not chunks:
        print("❌ No chunks found.")
        return

    lengths = [len(c["text"]) for c in chunks]

    print(f"Total chunks:        {len(chunks):,}")
    print(f"Average chars:       {sum(lengths) / len(lengths):,.1f}")
    print(f"Minimum chars:       {min(lengths):,}")
    print(f"Maximum chars:       {max(lengths):,}")

    sorted_lengths = sorted(lengths)

    def percentile(values, p):
        index = int(len(values) * p)
        index = min(index, len(values) - 1)
        return values[index]

    print(f"P50 chars:           {percentile(sorted_lengths, 0.50):,}")
    print(f"P90 chars:           {percentile(sorted_lengths, 0.90):,}")
    print(f"P95 chars:           {percentile(sorted_lengths, 0.95):,}")

    print("\nLength distribution:")

    buckets = [
        (0, 99),
        (100, 199),
        (200, 499),
        (500, 999),
        (1000, 1999),
        (2000, 4999),
        (5000, float("inf")),
    ]

    for low, high in buckets:
        count = sum(
            low <= length <= high
            for length in lengths
        )

        if high == float("inf"):
            label = f"{low:,}+"
        else:
            label = f"{low:,}-{high:,}"

        percentage = count / len(chunks) * 100

        print(f"  {label:12} {count:6,} ({percentage:5.1f}%)")


# ---------------------------------------------------------------------------
# Tiny chunks
# ---------------------------------------------------------------------------

def test_small_chunks(chunks):
    print_separator("2. SMALL / SUSPICIOUS CHUNKS")

    very_small = [
        c for c in chunks
        if len(c["text"].strip()) < VERY_SMALL_CHUNK_CHARS
    ]

    small = [
        c for c in chunks
        if len(c["text"].strip()) < MIN_CHUNK_CHARS
    ]

    print(
        f"Very small (< {VERY_SMALL_CHUNK_CHARS} chars): "
        f"{len(very_small):,}"
    )

    print(
        f"Small (< {MIN_CHUNK_CHARS} chars): "
        f"{len(small):,}"
    )

    if not small:
        print("✅ No suspiciously small chunks.")
        return

    print("\nExamples:")

    for i, chunk in enumerate(small[:20], start=1):
        metadata = chunk["metadata"]

        print(f"\n--- Small chunk {i} ---")
        print(
            f"chunk_id:    {metadata.get('chunk_id')}"
        )
        print(
            f"section_id:  {metadata.get('section_id')}"
        )
        print(
            f"section:     {metadata.get('heading')}"
        )
        print(
            f"characters:  {len(chunk['text'])}"
        )
        print(
            f"text:        {chunk['text'][:300]!r}"
        )


# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------

def test_metadata(chunks):
    print_separator("3. METADATA QUALITY")

    missing_counts = Counter()

    for chunk in chunks:
        metadata = chunk["metadata"]

        for key in REQUIRED_METADATA:
            value = metadata.get(key)

            if value is None or value == "":
                missing_counts[key] += 1

    if not missing_counts:
        print("✅ All required metadata fields are populated.")
    else:
        print("⚠️ Missing metadata:")

        for key, count in missing_counts.items():
            percentage = count / len(chunks) * 100

            print(
                f"  {key:20} "
                f"{count:,} ({percentage:.1f}%)"
            )


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def test_duplicates(chunks):
    print_separator("4. DUPLICATE DETECTION")

    content_hashes = Counter()
    chunk_ids = Counter()

    for chunk in chunks:
        text = normalize(chunk["text"])

        content_hashes[text] += 1

        chunk_id = chunk["metadata"].get("chunk_id")

        if chunk_id:
            chunk_ids[chunk_id] += 1

    duplicate_content = {
        text: count
        for text, count in content_hashes.items()
        if count > 1 and text
    }

    duplicate_ids = {
        chunk_id: count
        for chunk_id, count in chunk_ids.items()
        if count > 1
    }

    print(
        f"Duplicate content groups: {len(duplicate_content):,}"
    )

    print(
        f"Duplicate chunk IDs:      {len(duplicate_ids):,}"
    )

    if not duplicate_content and not duplicate_ids:
        print("✅ No duplicate content or IDs found.")
        return

    if duplicate_ids:
        print("\nDuplicate IDs:")

        for chunk_id, count in list(duplicate_ids.items())[:20]:
            print(f"  {chunk_id}: {count} occurrences")

    if duplicate_content:
        print("\nDuplicate content examples:")

        for text, count in list(duplicate_content.items())[:10]:
            print(f"\n  Occurrences: {count}")
            print(f"  Text: {text[:300]!r}")


# ---------------------------------------------------------------------------
# Section distribution
# ---------------------------------------------------------------------------

def test_section_distribution(chunks):
    print_separator("5. SECTION DISTRIBUTION")

    sections = Counter(
        metadata_value(c, "section_id", "UNKNOWN")
        for c in chunks
    )

    section_types = Counter(
        metadata_value(c, "section_type", "UNKNOWN")
        for c in chunks
    )

    print("Section types:")

    for section_type, count in section_types.most_common():
        print(f"  {section_type:30} {count:,}")

    print("\nLargest sections:")

    for section, count in sections.most_common(20):
        print(f"  {section:40} {count:,}")


# ---------------------------------------------------------------------------
# Inspect individual chunks
# ---------------------------------------------------------------------------

def inspect_sample(chunks, sample_size=10):
    print_separator(
        f"6. RANDOM SAMPLE ({sample_size} CHUNKS)"
    )

    if not chunks:
        return

    # Use evenly distributed samples instead of always getting
    # the first chunks, which can bias inspection toward the preamble.
    step = max(1, len(chunks) // sample_size)

    sample = chunks[::step][:sample_size]

    for i, chunk in enumerate(sample, start=1):
        metadata = chunk["metadata"]

        print(f"\n--- Chunk {i} ---")
        print(f"chunk_id:    {metadata.get('chunk_id')}")
        print(f"section_id:  {metadata.get('section_id')}")
        print(f"section:     {metadata.get('heading')}")
        print(f"section_type:{metadata.get('section_type')}")
        print(f"pages:       {metadata.get('page_start')} - "
              f"{metadata.get('page_end')}")
        print(f"characters:  {len(chunk['text'])}")

        print("\nText:")
        print(chunk["text"][:1000])


# ---------------------------------------------------------------------------
# Retrieval testing
# ---------------------------------------------------------------------------

def test_retrieval(vs, tests, top_k=5):
    print_separator("7. RETRIEVAL QUALITY TESTS")

    for test in tests:
        question = test["question"]
        expected = test.get("expected", [])

        print("\n" + "-" * 80)
        print(f"QUESTION: {question}")
        print("-" * 80)

        try:
            results = vs.retrieve(
                question,
                top_k=top_k
            )
        except Exception as exc:
            print(f"❌ Retrieval failed: {exc}")
            continue

        if not results:
            print("❌ No results returned.")
            continue

        found_expected = False

        for rank, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            text = result.get("text", "")

            section_id = metadata.get("section_id", "")
            heading = metadata.get("heading", "")
            chunk_id = metadata.get("chunk_id", "")

            combined = normalize(
                f"{section_id} {heading} {text}"
            )

            matches = [
                term
                for term in expected
                if normalize(term) in combined
            ]

            if matches:
                found_expected = True

            print(f"\n[{rank}]")
            print(f"section_id: {section_id}")
            print(f"chunk_id:   {chunk_id}")
            print(f"heading:    {heading}")

            if matches:
                print(f"✅ Expected match: {matches}")

            print(f"text: {text[:500].replace(chr(10), ' ')}")

        if expected:
            if found_expected:
                print("\n✅ Expected content found in top results.")
            else:
                print(
                    "\n⚠️ Expected content was NOT found "
                    "in the top results."
                )


# ---------------------------------------------------------------------------
# Article-specific retrieval test
# ---------------------------------------------------------------------------

def test_article_5(vs, top_k=5):
    print_separator("8. ARTICLE 5 FOCUSED TEST")

    question = "What does Article 5 say about prohibited practices?"

    print(f"Question: {question}\n")

    results = vs.retrieve(
        question,
        top_k=top_k
    )

    for rank, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        text = result.get("text", "")

        print(f"--- Result {rank} ---")
        print(
            f"section_id: {metadata.get('section_id')}"
        )
        print(
            f"chunk_id:   {metadata.get('chunk_id')}"
        )
        print(
            f"heading:    {metadata.get('heading')}"
        )
        print(
            f"pages:      {metadata.get('page_start')} - "
            f"{metadata.get('page_end')}"
        )
        print(f"text:\n{text[:1000]}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def inspect_chunks():
    print("=" * 80)
    print("EU AI ACT — CHUNK QUALITY INSPECTOR")
    print("=" * 80)

    print("\nLoading vector store...")

    vs = VectorStore(
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        persist_directory=CHROMA_DB_PATH,
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print(
        f"Chroma count: {vs.collection.count():,}"
    )

    print("\nLoading chunks...")

    chunks = get_all_chunks(vs)

    print(
        f"Loaded chunks: {len(chunks):,}"
    )

    # Run quality checks.
    test_basic_statistics(chunks)
    test_small_chunks(chunks)
    test_metadata(chunks)
    test_duplicates(chunks)
    test_section_distribution(chunks)

    # Human-readable inspection.
    inspect_sample(chunks, sample_size=10)

    # Retrieval evaluation.
    test_retrieval(
        vs,
        RETRIEVAL_TESTS,
        top_k=5,
    )

    # Focused test for the question you're currently using.
    test_article_5(vs, top_k=5)

    print_separator("DONE")

    print(
        "Review especially the small-chunk and retrieval sections."
    )


if __name__ == "__main__":
    inspect_chunks()