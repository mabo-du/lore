from pathlib import Path
from typing import List, Tuple

# We import these inside the functions to avoid massive load times or crashes if missing


class DiarizationEngine:
    """
    ONNX-based speaker diarization.

    Silero VAD → WeSpeaker embedding → spectral clustering.
    No PyTorch dependency. Replaces the old Resemblyzer and Pyannote 3.1 backends.
    """

    def __init__(self, num_speakers: int = 2):
        self.num_speakers = num_speakers

    def run_diarization(self, audio_path: Path) -> List[Tuple[float, float, str]]:
        """
        Run ONNX diarization on the given audio file.

        Returns a list of (start_s, end_s, speaker_label) tuples.
        """
        return self._run_onnx(audio_path)

    def _run_onnx(self, audio_path: Path) -> List[Tuple[float, float, str]]:
        """
        Pure ONNX diarization path with two-stage overlap-aware clustering.

        Stage 1: Extract clean embeddings from non-overlapping speech regions
                 and cluster to establish speaker centroids.
        Stage 2: Assign overlapping speech regions to the nearest centroid
                 (best-guess single speaker label).

        No PyTorch dependency.
        """
        from utils.model_manager import ModelManager
        from lore_core.vad import SileroVAD
        from lore_core.embedding import SpeakerEmbedding
        from lore_core.clustering import SpeakerClustering
        from lore_core.overlap_detector import OverlapDetector
        import wave
        import numpy as np
        from pathlib import Path

        # Read audio
        with wave.open(str(audio_path), "rb") as wf:
            rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        # ── Stage 1: VAD to find speech regions ──────────────────────────
        vad_model_dir = ModelManager.ensure_model("VAD")
        vad = SileroVAD()
        vad.load(Path(vad_model_dir) / ModelManager.VAD_FILENAME)
        speech_regions = vad.detect_speech_regions(audio)

        # ── Stage 2: Overlap detection ────────────────────────────────────
        overlap_regions = OverlapDetector().detect(audio_path)

        OVERLAP_RATIO_THRESHOLD = 0.5  # >50% overlap → treat as overlap region

        def _overlap_ratio(seg_start_s: float, seg_end_s: float) -> float:
            """Compute fraction of this segment that overlaps with any OverlapRegion."""
            seg_start_ms = int(seg_start_s * 1000)
            seg_end_ms = int(seg_end_s * 1000)
            seg_duration = max(seg_end_ms - seg_start_ms, 1)
            total_overlap_ms = 0
            for reg in overlap_regions:
                lo = max(seg_start_ms, reg.start_ms)
                hi = min(seg_end_ms, reg.end_ms)
                if lo < hi:
                    total_overlap_ms += hi - lo
            return total_overlap_ms / seg_duration

        # ── Stage 3: Extract embeddings, separate clean vs overlap ────────
        we_model_dir = ModelManager.ensure_model("WeSpeaker")
        embedder = SpeakerEmbedding()
        embedder.load(Path(we_model_dir) / "onnx/model.onnx")

        clean_embeddings = []  # For clustering
        clean_times = []       # Parallel list of (start_s, end_s)
        overlap_embeddings = []  # For nearest-centroid assignment
        overlap_times = []

        for start_sample, end_sample in speech_regions:
            segment = audio[start_sample:end_sample]
            if len(segment) < 1600:  # Skip <100ms
                continue

            start_s = start_sample / rate
            end_s = end_sample / rate
            emb = embedder.extract(segment)

            if _overlap_ratio(start_s, end_s) > OVERLAP_RATIO_THRESHOLD:
                overlap_embeddings.append(emb)
                overlap_times.append((start_s, end_s))
            else:
                clean_embeddings.append(emb)
                clean_times.append((start_s, end_s))

        if not clean_embeddings and not overlap_embeddings:
            return [(0.0, len(audio) / rate, "SPEAKER_00")]

        # ── Stage 4: Cluster clean embeddings ────────────────────────────
        all_labels = {}  # (start_s, end_s) → speaker_label

        if clean_embeddings:
            clusterer = SpeakerClustering(n_speakers=self.num_speakers)
            clean_labels = clusterer.cluster(np.array(clean_embeddings))

            # Compute cluster centroids from clean assignments
            n_clusters = max(clean_labels) + 1
            centroids = np.zeros((n_clusters, clean_embeddings[0].shape[0]), dtype=np.float32)
            counts = np.zeros(n_clusters, dtype=np.int32)
            for emb, label in zip(clean_embeddings, clean_labels):
                centroids[label] += emb
                counts[label] += 1
            for i in range(n_clusters):
                if counts[i] > 0:
                    centroids[i] /= counts[i]

            # Assign clean segments
            for (start_s, end_s), label in zip(clean_times, clean_labels):
                all_labels[(start_s, end_s)] = f"SPEAKER_{label:02d}"

            # ── Stage 5: Assign overlapping segments to nearest centroid ────
            for emb, (start_s, end_s) in zip(overlap_embeddings, overlap_times):
                # Find nearest centroid by cosine distance
                emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
                centroid_dists = []
                for centroid in centroids:
                    c_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
                    dist = 1.0 - np.dot(emb_norm, c_norm)  # cosine distance
                    centroid_dists.append(dist)
                nearest = int(np.argmin(centroid_dists))
                all_labels[(start_s, end_s)] = f"SPEAKER_{nearest:02d}"
        else:
            # No clean segments — cluster everything (fallback)
            all_embs = np.array(overlap_embeddings)
            clusterer = SpeakerClustering(n_speakers=self.num_speakers)
            labels = clusterer.cluster(all_embs)
            for (start_s, end_s), label in zip(overlap_times, labels):
                all_labels[(start_s, end_s)] = f"SPEAKER_{label:02d}"

        # ── Stage 6: Build results with contiguous merging ──────────────
        sorted_times = sorted(all_labels.keys(), key=lambda t: t[0])
        results = []
        for start_s, end_s in sorted_times:
            speaker = all_labels[(start_s, end_s)]
            if (
                results
                and results[-1][2] == speaker
                and (start_s - results[-1][1] < 0.5)
            ):
                results[-1] = (results[-1][0], end_s, speaker)
            else:
                results.append((start_s, end_s, speaker))

        return results

    @staticmethod
    def align_speakers_to_segments(
        segments: List["Segment"],  # noqa: F821
        diarization_results: List[Tuple[float, float, str]],
    ) -> None:
        """
        Aligns coarse speaker turns to fine-grained ASR segments.
        Modifies the 'speaker_label' property of the provided Segments in place.
        """
        if not diarization_results:
            return

        for seg in segments:
            seg_mid = (seg.start_ms + seg.end_ms) / 2000.0  # Midpoint in seconds

            # Find which diarization segment contains this midpoint
            assigned_speaker = None
            for d_start, d_end, speaker in diarization_results:
                if d_start <= seg_mid <= d_end:
                    assigned_speaker = speaker
                    break

            # If the midpoint falls in a gap, find the closest segment
            if not assigned_speaker:
                closest_speaker = None
                min_dist = float("inf")
                for d_start, d_end, speaker in diarization_results:
                    dist = min(abs(d_start - seg_mid), abs(d_end - seg_mid))
                    if dist < min_dist:
                        min_dist = dist
                        closest_speaker = speaker
                assigned_speaker = closest_speaker

            seg.speaker_label = assigned_speaker
