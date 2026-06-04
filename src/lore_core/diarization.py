from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

# We import these inside the functions to avoid massive load times or crashes if missing

class DiarizationEngine:
    def __init__(self, use_pyannote: bool = False, hf_token: Optional[str] = None):
        self.use_pyannote = use_pyannote
        self.hf_token = hf_token

    def run_diarization(self, audio_path: Path) -> List[Tuple[float, float, str]]:
        """
        Runs diarization on the given audio file.
        Returns a list of (start_s, end_s, speaker_label) tuples.
        """
        if self.use_pyannote:
            return self._run_pyannote(audio_path)
        else:
            return self._run_resemblyzer(audio_path)

    def _run_pyannote(self, audio_path: Path) -> List[Tuple[float, float, str]]:
        if not self.hf_token:
            raise ValueError("HuggingFace token is required for Pyannote.")
            
        import torch
        from pyannote.audio import Pipeline
        
        # Initialize pipeline (loads from HuggingFace cache or downloads if missing)
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.hf_token
        )
        
        if pipeline is None:
            raise RuntimeError("Failed to load Pyannote pipeline. Check your token and license acceptance.")
            
        # Send to CPU (or GPU if available, but we'll stick to CPU for reliability)
        pipeline.to(torch.device("cpu"))
        
        diarization = pipeline(str(audio_path))
        
        results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            results.append((turn.start, turn.end, speaker))
            
        return results

    def _run_resemblyzer(self, audio_path: Path) -> List[Tuple[float, float, str]]:
        """
        Lightweight, ungated fallback using Resemblyzer.
        Since Resemblyzer is just an embedding extractor, we cluster the embeddings manually.
        """
        from resemblyzer import VoiceEncoder, preprocess_wav
        from sklearn.cluster import KMeans
        
        encoder = VoiceEncoder("cpu")
        wav = preprocess_wav(str(audio_path))
        
        # Extract embeddings using a sliding window. 
        _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
        
        if len(cont_embeds) == 0:
            return [(0.0, len(wav)/16000.0, "SPEAKER_00")]
            
        # We assume 2 speakers for oral history by default in Resemblyzer fallback
        clusterer = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = clusterer.fit_predict(cont_embeds)
        
        results = []
        # Reconstruct timestamps based on wav_splits which are in samples
        sample_rate = 16000
        for i, split in enumerate(wav_splits):
            start_s = split.start / sample_rate
            end_s = split.stop / sample_rate
            speaker = f"SPEAKER_{labels[i]:02d}"
            
            # Merge contiguous segments with same speaker
            if results and results[-1][2] == speaker and (start_s - results[-1][1] < 0.5):
                results[-1] = (results[-1][0], end_s, speaker)
            else:
                results.append((start_s, end_s, speaker))
                
        return results

    @staticmethod
    def align_speakers_to_segments(segments: List['Segment'], diarization_results: List[Tuple[float, float, str]]) -> None:
        """
        Aligns coarse speaker turns to fine-grained ASR segments.
        Modifies the 'speaker_label' property of the provided Segments in place.
        """
        if not diarization_results:
            return
            
        for seg in segments:
            seg_mid = (seg.start_ms + seg.end_ms) / 2000.0 # Midpoint in seconds
            
            # Find which diarization segment contains this midpoint
            assigned_speaker = None
            for d_start, d_end, speaker in diarization_results:
                if d_start <= seg_mid <= d_end:
                    assigned_speaker = speaker
                    break
                    
            # If the midpoint falls in a gap, find the closest segment
            if not assigned_speaker:
                closest_speaker = None
                min_dist = float('inf')
                for d_start, d_end, speaker in diarization_results:
                    dist = min(abs(d_start - seg_mid), abs(d_end - seg_mid))
                    if dist < min_dist:
                        min_dist = dist
                        closest_speaker = speaker
                assigned_speaker = closest_speaker
                
            seg.speaker_label = assigned_speaker
