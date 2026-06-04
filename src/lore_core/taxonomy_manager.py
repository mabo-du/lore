import sqlite3
import sqlite_vec
import struct
from typing import List, Optional, Tuple, Dict, Any
from fastembed import TextEmbedding
from platformdirs import user_data_dir
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TaxonomyManager:
    """
    Manages taxonomy packs and vector search using sqlite-vec and FastEmbed.
    Supports hierarchical SKOS terms (preferred_term, broader_term, related_terms).
    """
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or ":memory:"
        self.embedding_model = None
        # We use a 384-dimensional model natively supported by FastEmbed
        self.dimension = 384
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        
        with self.conn:
            # Create vector table using vec0
            self.conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_taxonomy USING vec0(
                    id INTEGER PRIMARY KEY,
                    vector float[{self.dimension}]
                );
            """)
            # Create metadata table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS taxonomy_metadata (
                    id INTEGER PRIMARY KEY,
                    preferred_term TEXT NOT NULL,
                    definition TEXT,
                    broader_term TEXT,
                    related_terms TEXT
                );
            """)

    def _ensure_model(self):
        if self.embedding_model is None:
            logger.info("Loading FastEmbed model for taxonomy embedding...")
            cache_dir = Path(user_data_dir("lore", "lore_app")) / "fastembed"
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Using natively supported multilingual model in FastEmbed (replaces IBM Granite which requires custom ONNX loading)
            self.embedding_model = TextEmbedding(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                threads=1,
                cache_dir=str(cache_dir)
            )

    def serialize_f32(self, vector: List[float]) -> bytes:
        return struct.pack("%sf" % len(vector), *vector)

    def embed_text(self, text: str) -> List[float]:
        self._ensure_model()
        # FastEmbed returns a generator of numpy arrays
        return next(self.embedding_model.embed([text])).tolist()

    def add_term(self, preferred_term: str, definition: Optional[str] = None, broader_term: Optional[str] = None, related_terms: Optional[str] = None):
        """
        Adds a new taxonomy term. The text is concatenated for a denser semantic target.
        """
        components = [preferred_term]
        if definition: components.append(definition)
        if broader_term: components.append(f"Broader: {broader_term}")
        if related_terms: components.append(f"Related: {related_terms}")
        full_text = " | ".join(components)
        
        vector = self.embed_text(full_text)
        vector_bytes = self.serialize_f32(vector)
        
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO taxonomy_metadata (preferred_term, definition, broader_term, related_terms)
                VALUES (?, ?, ?, ?)
            """, (preferred_term, definition, broader_term, related_terms))
            row_id = cursor.lastrowid
            
            cursor.execute("""
                INSERT INTO vec_taxonomy (id, vector)
                VALUES (?, ?)
            """, (row_id, vector_bytes))

    def query(self, text: str, k: int = 3, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Queries the taxonomy for nearest matching terms based on cosine distance.
        Note: distance is returned by sqlite-vec (smaller is better).
        """
        vector = self.embed_text(text)
        vector_bytes = self.serialize_f32(vector)
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 
                tm.preferred_term, 
                tm.definition,
                distance
            FROM vec_taxonomy vt
            JOIN taxonomy_metadata tm ON vt.id = tm.id
            WHERE vector MATCH ? AND k = ?
            ORDER BY distance
        """, (vector_bytes, k))
        
        results = []
        for row in cursor.fetchall():
            distance = row[2]
            if distance <= threshold:
                results.append({
                    "preferred_term": row[0],
                    "definition": row[1],
                    "distance": distance
                })
        return results

    def close(self):
        if self.conn:
            self.conn.close()
