from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Segment:
    """Represents a discrete segment of transcribed audio.
    
    Rules:
        - confidence fields are Optional because SYSTEM-generated segments
          (silence markers, audio events) won't have logprob data.
    """
    start_ms: int
    end_ms: int
    text: str
    speaker_label: Optional[str] = None
    entities: list['EntityData'] = field(default_factory=list)
    tags: list['TagData'] = field(default_factory=list)
    # Anomaly detection fields (populated from faster-whisper diagnostics)
    avg_logprob: Optional[float] = None
    compression_ratio: Optional[float] = None
    no_speech_prob: Optional[float] = None
    
    # Translation field
    translation: Optional[str] = None
    
    # Word-level timestamp and probability data
    words: list[dict] = field(default_factory=list)

    # Backchannel detection
    is_backchannel: bool = False
    backchannel_source: str = ""  # "rule" | "acoustic" | ""
    
    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def confidence_level(self) -> str:
        """Returns a categorical confidence level for UI rendering.
        
        Returns one of: 'high', 'medium', 'low', 'hallucination', 'non_speech', or 'unknown'.
        Thresholds sourced from Whisper's internal fallback logic and research papers.
        """
        if self.avg_logprob is None:
            return "unknown"
        
        # Repetitive hallucination detection (highest priority)
        if self.compression_ratio is not None and self.compression_ratio > 2.4:
            return "hallucination"
        
        # Non-speech detection
        if self.no_speech_prob is not None and self.no_speech_prob > 0.6:
            return "non_speech"
        
        # Logprob-based confidence tiers
        if self.avg_logprob < -1.0:
            return "low"
        elif self.avg_logprob < -0.7:
            return "medium"
        else:
            return "high"

@dataclass
class EntityData:
    """Represents a named entity identified in text."""
    start_char: int
    end_char: int
    text: str
    label: str
    score: float
    segment_start_ms: int # Used to map back to the correct segment

@dataclass
class TagData:
    """Represents a domain taxonomy tag applied to a segment."""
    preferred_term: str
    definition: Optional[str]
    distance: float # Cosine distance from the segment vector
    segment_start_ms: int

@dataclass
class TranscriptMetadata:
    """Metadata about the transcript as a whole, decoupled from OHMS specific fields."""
    language: str = "en"
    duration_ms: int = 0
    model_used: str = "unknown"
    target_language: Optional[str] = None

@dataclass
class OverlapRegion:
    """A time region where two or more speakers were active simultaneously."""
    start_ms: int
    end_ms: int
    confidence: float  # Overlap probability from the segmentation model
    speakers: Optional[set[str]] = None  # Populated in Phase 3 (full attribution)


@dataclass
class Transcript:
    """The core document model."""
    segments: list[Segment] = field(default_factory=list)
    metadata: TranscriptMetadata = field(default_factory=TranscriptMetadata)
    overlap_regions: list[OverlapRegion] = field(default_factory=list)
