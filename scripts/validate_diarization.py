"""
VoxConverse DER validation script.

Compares Resemblyzer, Pyannote 3.1, and ONNX diarization paths
against ground-truth RTTM annotations.

Usage:
    python scripts/validate_diarization.py --audio-dir /path/to/wavs --rttm-dir /path/to/rttms

    Audio and RTTM files are matched by base filename (e.g. abjxc.wav ↔ abjxc.rttm).
    Results are printed as a table and saved to validation_results.json.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

# ─── DER computation using pyannote.metrics ──────────────────────────────────

def load_rttm(path: Path) -> list:
    """Load an RTTM file and return list of (start, end, speaker) tuples."""
    segments = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            # RTTM format: SPEAKER <file_id> <channel> <start> <duration> ... <speaker_label>
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            segments.append((start, start + duration, speaker))
    return segments


def segments_to_rttm_lines(segments: List[Tuple[float, float, str]], file_id: str) -> list:
    """Convert (start, end, speaker) tuples to RTTM format lines."""
    lines = []
    for start, end, speaker in segments:
        duration = end - start
        lines.append(
            f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
        )
    return lines


def compute_der(reference: list, hypothesis: list) -> dict:
    """
    Compute Diarization Error Rate and its components using pyannote.metrics.
    
    Returns dict with: der, confusion, miss, false_alarm, speaker_count_ref, speaker_count_hyp
    """
    from pyannote.core import Segment, Annotation
    from pyannote.metrics.diarization import DiarizationErrorRate
    
    # Convert to pyannote formats
    ref = Annotation()
    for start, end, speaker in reference:
        ref[Segment(start, end)] = speaker
    
    hyp = Annotation()
    for start, end, speaker in hypothesis:
        hyp[Segment(start, end)] = speaker
    
    metric = DiarizationErrorRate()
    der_value = metric(ref, hyp)
    
    # Extract components
    detail = metric.compute_components(ref, hyp)
    
    return {
        "der": float(der_value),
        "confusion": float(detail.get("confusion", 0)),
        "miss": float(detail.get("miss", 0)),
        "false_alarm": float(detail.get("false alarm", 0)),
        "speaker_count_ref": len(set(s[2] for s in reference)),
        "speaker_count_hyp": len(set(s[2] for s in hypothesis)),
    }


# ─── Diarization runners ────────────────────────────────────────────────────

def run_resemblyzer(audio_path: Path, num_speakers: int = 2) -> list:
    """Run Lore's current Resemblyzer diarization path."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from lore_core.diarization import DiarizationEngine
    engine = DiarizationEngine(use_pyannote=False, num_speakers=num_speakers)
    return engine.run_diarization(audio_path)


def run_pyannote(audio_path: Path, hf_token: str, num_speakers: int = 2) -> list:
    """Run Lore's current Pyannote 3.1 diarization path."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from lore_core.diarization import DiarizationEngine
    engine = DiarizationEngine(use_pyannote=True, hf_token=hf_token, num_speakers=num_speakers)
    return engine.run_diarization(audio_path)


def run_onnx(audio_path: Path, num_speakers: int = 2) -> list:
    """Run Lore's ONNX diarization path (Silero VAD + WeSpeaker + clustering)."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from lore_core.diarization import DiarizationEngine
    engine = DiarizationEngine(use_onnx=True, num_speakers=num_speakers)
    return engine.run_diarization(audio_path)


# ─── Main validation loop ────────────────────────────────────────────────────

def validate_file(
    audio_path: Path, rttm_path: Path, hf_token: str = "", num_speakers: int = 2
) -> dict:
    """Run all three diarization paths on one file and return DER results."""
    file_id = audio_path.stem
    reference = load_rttm(rttm_path)
    
    results = {"file": file_id, "reference_speakers": len(set(s[2] for s in reference))}
    
    # Resemblyzer
    try:
        hyp = run_resemblyzer(audio_path, num_speakers)
        results["resemblyzer"] = compute_der(reference, hyp)
    except Exception as e:
        results["resemblyzer"] = {"error": str(e)}
    
    # Pyannote (requires HF token)
    if hf_token:
        try:
            hyp = run_pyannote(audio_path, hf_token, num_speakers)
            results["pyannote"] = compute_der(reference, hyp)
        except Exception as e:
            results["pyannote"] = {"error": str(e)}
    
    # ONNX
    try:
        hyp = run_onnx(audio_path, num_speakers)
        results["onnx"] = compute_der(reference, hyp)
    except Exception as e:
        results["onnx"] = {"error": str(e)}
    
    return results


def main():
    parser = argparse.ArgumentParser(description="VoxConverse DER validation")
    parser.add_argument("--audio-dir", required=True, help="Directory of WAV files")
    parser.add_argument("--rttm-dir", required=True, help="Directory of RTTM files")
    parser.add_argument("--hf-token", default="", help="HuggingFace token for Pyannote path")
    parser.add_argument("--num-speakers", type=int, default=2, help="Speaker count hint")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N files (0 = all)")
    parser.add_argument("--output", default="validation_results.json", help="Output JSON path")
    args = parser.parse_args()
    
    audio_dir = Path(args.audio_dir)
    rttm_dir = Path(args.rttm_dir)
    
    # Match audio files to RTTM files by base name
    wav_files = sorted(audio_dir.glob("*.wav"))
    results = []
    
    if args.limit > 0:
        wav_files = wav_files[:args.limit]
    
    print(f"\nValidating {len(wav_files)} files...")
    print(f"{'File':<20} {'Resemblyzer DER':<18} {'Pyannote DER':<18} {'ONNX DER':<18}")
    print("-" * 74)
    
    for wav in wav_files:
        rttm = rttm_dir / f"{wav.stem}.rttm"
        if not rttm.exists():
            print(f"{wav.stem:<20} {'SKIP (no RTTM)':<54}")
            continue
        
        file_result = validate_file(wav, rttm, args.hf_token, args.num_speakers)
        results.append(file_result)
        
        res = file_result.get("resemblyzer", {})
        pya = file_result.get("pyannote", {})
        onx = file_result.get("onnx", {})
        
        r_der = f"{res['der']:.2%}" if "der" in res else res.get("error", "?")
        p_der = f"{pya['der']:.2%}" if "der" in pya else pya.get("error", "N/A")
        o_der = f"{onx['der']:.2%}" if "der" in onx else onx.get("error", "?")
        
        print(f"{wav.stem:<20} {r_der:<18} {p_der:<18} {o_der:<18}")
    
    # Summary
    print("\n" + "=" * 74)
    for path_name in ["resemblyzer", "pyannote", "onnx"]:
        ders = [
            r[path_name]["der"]
            for r in results
            if path_name in r and "der" in r[path_name]
        ]
        if ders:
            avg = sum(ders) / len(ders)
            print(f"{path_name:<20} avg DER: {avg:.2%}  (over {len(ders)} files)")
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
