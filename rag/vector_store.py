"""
ChromaDB-backed vector store for the EU AI Act RAG system.

Responsibilities:
    - Load the embedding model.
    - Create/load a ChromaDB collection.
    - Store embeddings with stable IDs.
    - Add/upsert/delete documents.
    - Reset collections safely.
    - Retrieve semantically similar documents.
    - Return all documents for BM25/hybrid retrieval.
    - Report collection information.

Design notes:
    - Embeddings are generated locally with SentenceTransformers.
    - Embeddings are normalized.
    - Chroma is explicitly configured to use cosine distance.
    - Documents are inserted in batches so the entire embedding matrix
      does not need to remain in memory.
    - Collection metadata records the embedding model and corpus version.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, List, Optional, Sequence

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStore:
    """Vector database wrapper around ChromaDB."""

    DISTANCE_SPACE = "cosine"

    def __init__(
        self,
        collection_name: str,
        embedding_model: str,
        persist_directory: str = "./data/chromadb",
        batch_size: int = 64,
        collection_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not collection_name:
            raise ValueError("collection_name must not be empty.")

        if not embedding_model:
            raise ValueError("embedding_model must not be empty.")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.persist_directory = persist_directory
        self.batch_size = batch_size

        self.collection_metadata = {
            "embedding_model": self.embedding_model_name,
            "distance_space": self.DISTANCE_SPACE,
        }

        if collection_metadata:
            self.collection_metadata.update(
                collection_metadata
            )

        print(
            f"🧠 Loading embedding model: "
            f"{self.embedding_model_name}"
        )

        self.embedding_model = SentenceTransformer(
            self.embedding_model_name
        )

        print(
            f"🗄️ Initializing ChromaDB: "
            f"{self.persist_directory}"
        )

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
            ),
        )

        self.collection = self._get_or_create_collection()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _create_collection(self):
        """Create a new collection with explicit vector configuration."""

        return self.client.create_collection(
            name=self.collection_name,
            metadata=self.collection_metadata,
            configuration={
                "hnsw": {
                    "space": self.DISTANCE_SPACE,
                }
            },
        )

    def _get_or_create_collection(self):
        """Load an existing collection or create a new one."""

        existing_collections = self.client.list_collections()

        existing_names = {
            collection.name
            for collection in existing_collections
        }

        if self.collection_name in existing_names:
            collection = self.client.get_collection(
                name=self.collection_name
            )

            print(
                f"✅ Loaded existing collection: "
                f"{self.collection_name}"
            )

            self._validate_existing_collection(
                collection
            )

            print(
                f"📊 Existing documents: "
                f"{collection.count():,}"
            )

            return collection

        collection = self._create_collection()

        print(
            f"✅ Created new collection: "
            f"{self.collection_name}"
        )

        return collection

    def _validate_existing_collection(
        self,
        collection: Any,
    ) -> None:
        """
        Validate important properties of an existing collection.

        This prevents accidentally querying a collection created for a
        different embedding model.
        """

        metadata = collection.metadata or {}

        stored_model = metadata.get(
            "embedding_model"
        )

        if (
            stored_model
            and stored_model != self.embedding_model_name
        ):
            raise RuntimeError(
                "Embedding model mismatch for existing collection "
                f"'{self.collection_name}'. "
                f"Stored model: {stored_model!r}; "
                f"configured model: "
                f"{self.embedding_model_name!r}."
            )

        stored_space = metadata.get(
            "distance_space"
        )

        if (
            stored_space
            and stored_space != self.DISTANCE_SPACE
        ):
            raise RuntimeError(
                "Distance-space mismatch for existing collection "
                f"'{self.collection_name}'. "
                f"Stored space: {stored_space!r}; "
                f"expected: {self.DISTANCE_SPACE!r}."
            )

    def delete_collection(self) -> None:
        """
        Delete the collection completely.

        Unlike reset_collection(), this method does not recreate it.
        """

        try:
            self.client.delete_collection(
                name=self.collection_name
            )

            print(
                f"🗑️ Deleted collection: "
                f"{self.collection_name}"
            )

            self.collection = None

        except Exception as exc:
            print(
                f"⚠️ Could not delete collection "
                f"'{self.collection_name}': {exc}"
            )
            raise

    def reset_collection(self) -> None:
        """Delete the collection and recreate an empty one."""

        if self.collection_name in {
            collection.name
            for collection in self.client.list_collections()
        }:
            self.client.delete_collection(
                name=self.collection_name
            )

            print(
                f"🗑️ Deleted collection: "
                f"{self.collection_name}"
            )

        self.collection = self._create_collection()

        print(
            f"✅ Created empty collection: "
            f"{self.collection_name}"
        )

    def clear_collection(self) -> None:
        """Alias for reset_collection()."""

        self.reset_collection()

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def _encode_documents(
        self,
        texts: Sequence[str],
    ) -> List[List[float]]:
        """Generate normalized embeddings for documents."""

        if not texts:
            return []

        embeddings = self.embedding_model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def _encode_query(
        self,
        query: str,
    ) -> List[float]:
        """Generate a normalized embedding for a query."""

        embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.tolist()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_metadata(
        metadatas: Sequence[Dict[str, Any]],
    ) -> None:
        """Validate Chroma-compatible metadata values."""

        allowed_types = (
            str,
            int,
            float,
            bool,
        )

        for index, metadata in enumerate(metadatas):
            if not isinstance(metadata, dict):
                raise TypeError(
                    f"Metadata {index} must be a dictionary."
                )

            for key, value in metadata.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"Metadata key {key!r} must be a string."
                    )

                if value is None:
                    raise ValueError(
                        f"Metadata '{key}' at index {index} "
                        "cannot be None."
                    )

                if not isinstance(value, allowed_types):
                    raise TypeError(
                        f"Metadata '{key}' at index {index} "
                        f"has unsupported type "
                        f"{type(value).__name__}."
                    )

    @staticmethod
    def _prepare_documents(
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]],
        ids: Optional[Sequence[str]],
        collection_name: str,
    ) -> tuple[
        List[str],
        List[Dict[str, Any]],
        List[str],
    ]:
        """Validate and normalize documents, metadata, and IDs."""

        if not texts:
            return [], [], []

        cleaned_texts: List[str] = []

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"Document {index} is not a string."
                )

            cleaned = text.strip()

            if not cleaned:
                raise ValueError(
                    f"Document {index} is empty."
                )

            cleaned_texts.append(cleaned)

        if metadatas is None:
            cleaned_metadatas = [
                {"chunk_id": index}
                for index in range(len(cleaned_texts))
            ]
        else:
            cleaned_metadatas = [
                dict(metadata)
                for metadata in metadatas
            ]

        if len(cleaned_metadatas) != len(cleaned_texts):
            raise ValueError(
                "texts and metadatas must have the same length."
            )

        VectorStore._validate_metadata(
            cleaned_metadatas
        )

        if ids is None:
            generated_ids = [
                f"{collection_name}_doc_{index}"
                for index in range(len(cleaned_texts))
            ]
        else:
            generated_ids = [
                str(document_id)
                for document_id in ids
            ]

        if len(generated_ids) != len(cleaned_texts):
            raise ValueError(
                "texts and ids must have the same length."
            )

        if any(not document_id for document_id in generated_ids):
            raise ValueError(
                "Document IDs must not be empty."
            )

        if len(set(generated_ids)) != len(generated_ids):
            raise ValueError(
                "Document IDs must be unique."
            )

        return (
            cleaned_texts,
            cleaned_metadatas,
            generated_ids,
        )

    # ------------------------------------------------------------------
    # Stable IDs
    # ------------------------------------------------------------------

    @staticmethod
    def make_chunk_id(
        section_id: str,
        chunk_index: int,
    ) -> str:
        """
        Create a stable human-readable chunk ID.

        Example:
            article_5:chunk_2
        """

        if not section_id:
            raise ValueError(
                "section_id must not be empty."
            )

        if chunk_index < 0:
            raise ValueError(
                "chunk_index must be >= 0."
            )

        return (
            f"{section_id}:"
            f"chunk_{chunk_index}"
        )

    @staticmethod
    def make_content_hash(
        text: str,
    ) -> str:
        """Return a SHA-256 hash for document content."""

        return sha256(
            text.encode("utf-8")
        ).hexdigest()

    # ------------------------------------------------------------------
    # Add documents
    # ------------------------------------------------------------------

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        Add documents to ChromaDB.

        Documents are embedded and inserted batch-by-batch.
        """

        (
            cleaned_texts,
            cleaned_metadatas,
            document_ids,
        ) = self._prepare_documents(
            texts,
            metadatas,
            ids,
            self.collection_name,
        )

        if not cleaned_texts:
            print(
                "⚠️ No documents supplied. Nothing to add."
            )
            return

        total = len(cleaned_texts)

        print(
            f"🔢 Generating and storing embeddings "
            f"for {total:,} documents..."
        )

        for start in range(
            0,
            total,
            self.batch_size,
        ):
            end = min(
                start + self.batch_size,
                total,
            )

            batch_texts = cleaned_texts[start:end]

            embeddings = self._encode_documents(
                batch_texts
            )

            self.collection.add(
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=cleaned_metadatas[start:end],
                ids=document_ids[start:end],
            )

            print(
                f"   📥 Added "
                f"{end:,}/{total:,}"
            )

        print(
            f"✅ Added {total:,} documents."
        )

    # ------------------------------------------------------------------
    # Upsert documents
    # ------------------------------------------------------------------

    def upsert_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Add or update documents in batches."""

        (
            cleaned_texts,
            cleaned_metadatas,
            document_ids,
        ) = self._prepare_documents(
            texts,
            metadatas,
            ids,
            self.collection_name,
        )

        if not cleaned_texts:
            print(
                "⚠️ No documents supplied. Nothing to upsert."
            )
            return

        total = len(cleaned_texts)

        for start in range(
            0,
            total,
            self.batch_size,
        ):
            end = min(
                start + self.batch_size,
                total,
            )

            batch_texts = cleaned_texts[start:end]

            embeddings = self._encode_documents(
                batch_texts
            )

            self.collection.upsert(
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=cleaned_metadatas[start:end],
                ids=document_ids[start:end],
            )

            print(
                f"   🔄 Upserted "
                f"{end:,}/{total:,}"
            )

        print(
            f"✅ Upserted {total:,} documents."
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        max_distance: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most similar documents.

        Parameters
        ----------
        query:
            Natural-language search query.

        top_k:
            Maximum number of results.

        score_threshold:
            Backwards-compatible name for maximum cosine distance.

        max_distance:
            Preferred name for maximum cosine distance.
        where:
            Optional Chroma metadata filter.
        where_document:
            Optional Chroma document-content filter.

        Returns
        -------
        list[dict]
            Each result contains:
                id
                text
                distance
                metadata
        """

        if not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        if (
            score_threshold is not None
            and max_distance is not None
            and score_threshold != max_distance
        ):
            raise ValueError(
                "Specify only one of score_threshold "
                "or max_distance."
            )

        if max_distance is None:
            max_distance = score_threshold

        if max_distance is not None:
            if max_distance < 0:
                raise ValueError(
                    "max_distance must be >= 0."
                )

        collection_count = self.collection.count()

        if collection_count == 0:
            return []

        top_k = min(
            top_k,
            collection_count,
        )

        query_embedding = self._encode_query(
            query.strip()
        )

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": [
                "documents",
                "distances",
                "metadatas",
            ],
        }

        if where is not None:
            query_kwargs["where"] = where

        if where_document is not None:
            query_kwargs["where_document"] = where_document

        results = self.collection.query(**query_kwargs)

        documents = (
            results.get("documents")
            or [[]]
        )

        distances = (
            results.get("distances")
            or [[]]
        )

        metadatas = (
            results.get("metadatas")
            or [[]]
        )

        result_ids = (
            results.get("ids")
            or [[]]
        )

        if not documents or not documents[0]:
            return []

        retrieved_documents: List[
            Dict[str, Any]
        ] = []

        for index, document in enumerate(
            documents[0]
        ):
            distance = (
                distances[0][index]
                if distances
                and distances[0]
                and index < len(distances[0])
                else None
            )

            metadata = (
                metadatas[0][index]
                if metadatas
                and metadatas[0]
                and index < len(metadatas[0])
                else {}
            )

            document_id = (
                result_ids[0][index]
                if result_ids
                and result_ids[0]
                and index < len(result_ids[0])
                else None
            )

            if (
                max_distance is not None
                and distance is not None
                and distance > max_distance
            ):
                continue

            retrieved_documents.append(
                {
                    "id": document_id,
                    "text": document,
                    "distance": distance,
                    "similarity": (
                        1.0 - float(distance)
                        if distance is not None
                        else None
                    ),
                    "metadata": metadata,
                }
            )

        return retrieved_documents

    # ------------------------------------------------------------------
    # Collection information
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of documents."""

        return self.collection.count()

    def info(self) -> Dict[str, Any]:
        """Return basic vector store information."""

        metadata = self.collection.metadata or {}

        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "embedding_model": self.embedding_model_name,
            "persist_directory": self.persist_directory,
            "batch_size": self.batch_size,
            "distance_space": self.DISTANCE_SPACE,
            "collection_metadata": metadata,
        }

    def print_info(self) -> None:
        """Print vector store information."""

        info = self.info()

        print("\n📊 Vector Store")
        print(
            f"   Collection: "
            f"{info['collection_name']}"
        )
        print(
            f"   Documents: "
            f"{info['document_count']:,}"
        )
        print(
            f"   Embedding model: "
            f"{info['embedding_model']}"
        )
        print(
            f"   Distance space: "
            f"{info['distance_space']}"
        )
        print(
            f"   Storage: "
            f"{info['persist_directory']}"
        )

    # ------------------------------------------------------------------
    # Get documents
    # ------------------------------------------------------------------

    def get_all_documents(
        self,
        batch_size: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve every document, ID, and metadata.

        Useful for constructing BM25 indexes and hybrid retrieval.
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        total = self.collection.count()

        if total == 0:
            return []

        documents: List[Dict[str, Any]] = []

        for offset in range(
            0,
            total,
            batch_size,
        ):
            results = self.collection.get(
                limit=min(
                    batch_size,
                    total - offset,
                ),
                offset=offset,
                include=[
                    "documents",
                    "metadatas",
                ],
            )

            batch_ids = (
                results.get("ids")
                or []
            )

            batch_documents = (
                results.get("documents")
                or []
            )

            batch_metadatas = (
                results.get("metadatas")
                or []
            )

            for index, text in enumerate(
                batch_documents
            ):
                metadata = (
                    batch_metadatas[index]
                    if index < len(batch_metadatas)
                    else {}
                )

                document_id = (
                    batch_ids[index]
                    if index < len(batch_ids)
                    else None
                )

                documents.append(
                    {
                        "id": document_id,
                        "text": text,
                        "metadata": metadata,
                    }
                )

        return documents

    # ------------------------------------------------------------------
    # Raw collection access
    # ------------------------------------------------------------------

    def get(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Expose Chroma's get() for debugging/admin tasks."""

        kwargs: Dict[str, Any] = {}

        if limit is not None:
            kwargs["limit"] = limit

        if offset is not None:
            kwargs["offset"] = offset

        if include is not None:
            kwargs["include"] = include

        if ids is not None:
            kwargs["ids"] = ids

        if where is not None:
            kwargs["where"] = where

        return self.collection.get(**kwargs)

    # ------------------------------------------------------------------
    # Delete documents
    # ------------------------------------------------------------------

    def delete_documents(
        self,
        ids: List[str],
    ) -> None:
        """Delete specific documents by ID."""

        if not ids:
            return

        self.collection.delete(
            ids=ids
        )

        print(
            f"🗑️ Deleted {len(ids):,} documents."
        )
