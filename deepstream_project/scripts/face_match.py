"""
face_match.py — Face Matching Utility Module

Handles loading the local face database (.npy files) and performing
cosine similarity matching between live embeddings and stored embeddings.

Usage:
    from face_match import FaceDatabase

    db = FaceDatabase("/workspace/faces", threshold=0.7)
    name, score = db.find_best_match(live_embedding)
"""

import os
import sys
import numpy as np
from typing import Dict, Tuple, Optional


class FaceDatabase:
    """
    Manages a local database of face embeddings stored as .npy files.

    Each .npy file should contain a single 1D numpy array representing
    the face embedding vector (128D for MobileFaceNet).

    File naming convention:
        /workspace/faces/sharad.npy  →  identity = "sharad"
        /workspace/faces/rahul.npy   →  identity = "rahul"
    """

    def __init__(self, db_path: str, threshold: float = 0.7):
        """
        Initialize the face database.

        Args:
            db_path:   Path to directory containing .npy embedding files.
            threshold: Cosine similarity threshold for positive match.
                       MobileFaceNet recommended: 0.7
        """
        self.db_path = db_path
        self.threshold = threshold
        self.embeddings: Dict[str, np.ndarray] = {}
        self._load_database()

    def _load_database(self) -> None:
        """Load all .npy files from the database directory."""
        if not os.path.isdir(self.db_path):
            print(f"[FaceDB] WARNING: Database path '{self.db_path}' does not exist.")
            print(f"[FaceDB] Creating directory...")
            os.makedirs(self.db_path, exist_ok=True)
            return

        npy_files = [f for f in os.listdir(self.db_path) if f.endswith(".npy")]

        if not npy_files:
            print(f"[FaceDB] WARNING: No .npy files found in '{self.db_path}'.")
            print(f"[FaceDB] Register faces using: python scripts/register_face.py")
            return

        for filename in npy_files:
            filepath = os.path.join(self.db_path, filename)
            name = os.path.splitext(filename)[0]

            try:
                embedding = np.load(filepath)

                # Validate embedding shape
                if embedding.ndim != 1:
                    print(f"[FaceDB] WARNING: '{filename}' has shape {embedding.shape}, "
                          f"expected 1D. Flattening...")
                    embedding = embedding.flatten()

                if embedding.shape[0] not in (128, 256, 512):
                    print(f"[FaceDB] WARNING: '{filename}' has {embedding.shape[0]} dims, "
                          f"expected 128/256/512.")

                # Normalize the embedding for cosine similarity
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                self.embeddings[name] = embedding
                print(f"[FaceDB] Loaded: {name} ({embedding.shape[0]}D)")

            except Exception as e:
                print(f"[FaceDB] ERROR loading '{filename}': {e}")

        print(f"[FaceDB] Database loaded: {len(self.embeddings)} identities.")

    def reload(self) -> None:
        """Reload the face database (e.g., after registering new faces)."""
        self.embeddings.clear()
        self._load_database()

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec_a: First embedding vector (should be L2-normalized).
            vec_b: Second embedding vector (should be L2-normalized).

        Returns:
            Cosine similarity score in range [-1, 1].
        """
        # If vectors are already normalized, dot product = cosine similarity
        dot_product = np.dot(vec_a, vec_b)

        # Clamp to valid range to avoid floating point issues
        return float(np.clip(dot_product, -1.0, 1.0))

    def find_best_match(
        self, embedding: np.ndarray
    ) -> Tuple[str, float]:
        """
        Find the best matching identity for a given embedding.

        Args:
            embedding: Live face embedding vector (128D for MobileFaceNet).

        Returns:
            Tuple of (name, score):
                - name:  Identity name if match found, else "Unknown"
                - score: Cosine similarity score of best match
        """
        if not self.embeddings:
            return ("Unknown", 0.0)

        # Normalize the live embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        else:
            return ("Unknown", 0.0)

        best_name = "Unknown"
        best_score = -1.0

        for name, stored_embedding in self.embeddings.items():
            score = self.cosine_similarity(embedding, stored_embedding)

            if score > best_score:
                best_score = score
                best_name = name

        # Apply threshold check
        if best_score >= self.threshold:
            return (best_name, best_score)
        else:
            return ("Unknown", best_score)

    def get_all_matches(
        self, embedding: np.ndarray
    ) -> list:
        """
        Get similarity scores against all stored embeddings (for debugging).

        Args:
            embedding: Live face embedding vector.

        Returns:
            List of (name, score) tuples sorted by score (descending).
        """
        if not self.embeddings:
            return []

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        results = []
        for name, stored_embedding in self.embeddings.items():
            score = self.cosine_similarity(embedding, stored_embedding)
            results.append((name, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def __len__(self) -> int:
        return len(self.embeddings)

    def __repr__(self) -> str:
        names = list(self.embeddings.keys())
        return (f"FaceDatabase(path='{self.db_path}', "
                f"threshold={self.threshold}, "
                f"identities={names})")


# ─────────────────────── Standalone Test ────────────────────────────────────
if __name__ == "__main__":
    """Quick test with synthetic embeddings."""
    import tempfile

    print("=" * 60)
    print("  FaceDatabase — Unit Test with Synthetic Data")
    print("=" * 60)

    # Create temp directory with test embeddings
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two "known" faces
        emb_alice = np.random.randn(128).astype(np.float32)
        emb_bob = np.random.randn(128).astype(np.float32)
        np.save(os.path.join(tmpdir, "alice.npy"), emb_alice)
        np.save(os.path.join(tmpdir, "bob.npy"), emb_bob)

        # Initialize database
        db = FaceDatabase(tmpdir, threshold=0.7)
        print(f"\n{db}\n")

        # Test 1: Match with Alice's own embedding (should be perfect match)
        name, score = db.find_best_match(emb_alice)
        print(f"Test 1 — Alice vs DB: {name} (score={score:.4f})")
        assert name == "alice" and score > 0.99, "FAILED: Should match Alice"

        # Test 2: Match with Bob's own embedding
        name, score = db.find_best_match(emb_bob)
        print(f"Test 2 — Bob vs DB:   {name} (score={score:.4f})")
        assert name == "bob" and score > 0.99, "FAILED: Should match Bob"

        # Test 3: Match with random unknown face
        emb_unknown = np.random.randn(128).astype(np.float32)
        name, score = db.find_best_match(emb_unknown)
        print(f"Test 3 — Unknown:     {name} (score={score:.4f})")
        # Note: Random vectors may occasionally exceed threshold

        # Test 4: Get all matches
        all_matches = db.get_all_matches(emb_alice)
        print(f"Test 4 — All matches for Alice: {all_matches}")

        print("\n✅ All tests passed!")
