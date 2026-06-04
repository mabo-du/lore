import pytest
from lore_core.global_search import GlobalSearchIndex
from models.transcript import Transcript, Segment

@pytest.fixture
def search_index(tmp_path):
    # This might download BAAI/bge-small-en-v1.5 on the first run if not cached,
    # but since RAG Phase 13 used it, it should be cached locally.
    db_path = tmp_path / "test_search.db"
    return GlobalSearchIndex(db_path=db_path)

def test_index_and_search_keyword(search_index):
    # 1. Create a dummy transcript
    t1 = Transcript()
    t1.segments.append(Segment(start_ms=0, end_ms=1000, text="The quick brown fox jumps over the lazy dog."))
    
    t2 = Transcript()
    t2.segments.append(Segment(start_ms=0, end_ms=1000, text="Some unrelated text about space exploration."))
    
    # 2. Index them
    search_index.index_transcript("project_fox", t1)
    search_index.index_transcript("project_space", t2)
    
    # 3. Keyword Search
    results = search_index.search_keyword("brown fox")
    assert len(results) == 1
    assert results[0]["project_id"] == "project_fox"
    assert "<b>brown</b>" in results[0]["snippet"] or "<b>fox</b>" in results[0]["snippet"]

def test_index_and_search_semantic(search_index):
    t1 = Transcript()
    t1.segments.append(Segment(start_ms=0, end_ms=1000, text="The quick brown fox jumps over the lazy dog."))
    
    t2 = Transcript()
    t2.segments.append(Segment(start_ms=0, end_ms=1000, text="NASA launched a new rover to Mars today."))
    
    search_index.index_transcript("project_fox", t1)
    search_index.index_transcript("project_space", t2)
    
    # Semantic Search - "red planet" is semantically related to Mars, not fox
    results = search_index.search_semantic("red planet", limit=1)
    assert len(results) == 1
    assert results[0]["project_id"] == "project_space"
    
    # Re-indexing the same project should replace old records
    search_index.index_transcript("project_fox", t1)
    results = search_index.search_keyword("lazy dog")
    # We should still only have 1 hit, not 2
    assert len(results) == 1
