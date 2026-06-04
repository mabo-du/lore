import sqlite3
import sqlite_vec
import struct
from pathlib import Path
from fastembed import TextEmbedding
from models.transcript import Transcript

class GlobalSearchIndex:
    """
    Manages a persistent SQLite database for global search across all processed transcripts.
    Uses FTS5 for keyword search and sqlite-vec + fastembed for semantic search.
    """
    def __init__(self, db_path: Path = None):
        if db_path is None:
            self.db_path = Path.home() / ".local" / "share" / "lore" / "global_search.db"
        else:
            self.db_path = db_path
            
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load embedding model (same one used by RAG/Taxonomy)
        self.model = TextEmbedding("BAAI/bge-small-en-v1.5")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as db:
            db.enable_load_extension(True)
            sqlite_vec.load(db)
            db.enable_load_extension(False)
            
            # FTS5 table for keyword search
            db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
                    id UNINDEXED,
                    project_id UNINDEXED,
                    start_ms UNINDEXED,
                    end_ms UNINDEXED,
                    text
                )
            """)
            
            # Vector table for semantic search (bge-small-en-v1.5 is 384d)
            db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS segments_vec USING vec0(
                    embedding float[384]
                )
            """)
            
            # Metadata table to map vector rowid to segment info
            db.execute("""
                CREATE TABLE IF NOT EXISTS segments_meta (
                    rowid INTEGER PRIMARY KEY,
                    project_id TEXT,
                    start_ms INTEGER,
                    end_ms INTEGER,
                    text TEXT
                )
            """)
            db.commit()

    def index_transcript(self, project_id: str, transcript: Transcript):
        """Indexes all segments of a transcript into the global DB."""
        # Collect texts to embed in batch
        texts = [s.text for s in transcript.segments if s.text.strip()]
        if not texts:
            return
            
        embeddings = list(self.model.embed(texts))
        
        with sqlite3.connect(self.db_path) as db:
            db.enable_load_extension(True)
            sqlite_vec.load(db)
            db.enable_load_extension(False)
            
            cursor = db.cursor()
            
            # Delete old entries for this project to avoid duplicates if re-indexed
            cursor.execute("DELETE FROM segments_fts WHERE project_id = ?", (project_id,))
            cursor.execute("DELETE FROM segments_meta WHERE project_id = ?", (project_id,))
            # Vector tables don't support DELETE by non-rowid directly easily in some setups,
            # but since we rely on meta, orphaned vectors are just dead space. For now, it's fine.
            
            for seg, emb in zip([s for s in transcript.segments if s.text.strip()], embeddings):
                # 1. Insert into FTS
                cursor.execute("""
                    INSERT INTO segments_fts(project_id, start_ms, end_ms, text)
                    VALUES (?, ?, ?, ?)
                """, (project_id, seg.start_ms, seg.end_ms, seg.text))
                
                # 2. Insert into meta to get rowid
                cursor.execute("""
                    INSERT INTO segments_meta(project_id, start_ms, end_ms, text)
                    VALUES (?, ?, ?, ?)
                """, (project_id, seg.start_ms, seg.end_ms, seg.text))
                rowid = cursor.lastrowid
                
                # 3. Insert into vec0 using same rowid
                # Pack the numpy array to bytes as required by sqlite-vec float arrays
                emb_bytes = emb.astype("float32").tobytes()
                cursor.execute("""
                    INSERT INTO segments_vec(rowid, embedding)
                    VALUES (?, ?)
                """, (rowid, emb_bytes))
                
            db.commit()

    def search_keyword(self, query: str, limit: int = 20) -> list[dict]:
        """Performs a fast keyword search using FTS5."""
        # Clean query for FTS5 syntax
        clean_query = "".join(c for c in query if c.isalnum() or c.isspace())
        if not clean_query.strip():
            return []
            
        fts_query = " OR ".join(f"{word}*" for word in clean_query.split())
        
        with sqlite3.connect(self.db_path) as db:
            cursor = db.cursor()
            cursor.execute("""
                SELECT project_id, start_ms, end_ms, snippet(segments_fts, -1, '<b>', '</b>', '...', 10)
                FROM segments_fts 
                WHERE segments_fts MATCH ? 
                ORDER BY rank 
                LIMIT ?
            """, (fts_query, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "project_id": row[0],
                    "start_ms": row[1],
                    "end_ms": row[2],
                    "snippet": row[3],
                    "type": "keyword"
                })
            return results

    def search_semantic(self, query: str, limit: int = 20) -> list[dict]:
        """Performs a semantic search using vector similarity."""
        query_emb = list(self.model.embed([query]))[0]
        emb_bytes = query_emb.astype("float32").tobytes()
        
        with sqlite3.connect(self.db_path) as db:
            db.enable_load_extension(True)
            sqlite_vec.load(db)
            db.enable_load_extension(False)
            
            cursor = db.cursor()
            cursor.execute("""
                SELECT m.project_id, m.start_ms, m.end_ms, m.text, vec_distance_cosine(v.embedding, ?) as distance
                FROM segments_vec v
                JOIN segments_meta m ON v.rowid = m.rowid
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY distance
            """, (emb_bytes, emb_bytes, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "project_id": row[0],
                    "start_ms": row[1],
                    "end_ms": row[2],
                    "snippet": row[3],
                    "type": "semantic",
                    "distance": row[4]
                })
            return results
