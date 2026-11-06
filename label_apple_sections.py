#!/usr/bin/env python3
"""Label existing Apple sections with pretrained SongFormer; never change boundaries."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent
SONGFORMER_PATH = BASE / "vendor" / "SongFormer"
APPLE_PATH = BASE / "outputs.local" / "apple"
OUTPUT_PATH = BASE / "outputs.local" / "labeled"
DEVICE = "auto"  # auto = CUDA jika tersedia, selain itu CPU; "mps" opsional
ENCODER_WINDOW_SECONDS = 120  # sama dengan pipeline SongMapApp; head tetap 420 s
LIMIT = 5  # 0 = semua JSON Apple
LABELS = {0: "intro", 1: "verse", 2: "chorus", 3: "bridge",
          4: "inst", 5: "outro", 6: "silence", 26: "pre-chorus"}

def aggregate(sections, frames):
    """Duration-weighted class probabilities; frames = (start,end,probabilities)."""
    result = []
    for index, section in enumerate(sections):
        start, end = float(section["start"]), float(section["end"])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError("Invalid Apple section")
        scores = [0.0] * len(LABELS)
        covered = 0.0
        for left, right, probabilities in frames:
            overlap = max(0.0, min(end, right) - max(start, left))
            if overlap:
                covered += overlap
                scores = [a + overlap * b for a, b in zip(scores, probabilities)]
        if not covered:
            raise ValueError(f"No SongFormer coverage for section {index + 1}")
        scores = [v / covered for v in scores]
        order = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        names = list(LABELS.values())
        result.append({**section, "id": f"s{index + 1:04d}",
                       "label_initial": names[order[0]], "label_manual": None,
                       "label": names[order[0]], "review_status": "pending",
                       "label_scores": dict(zip(names, scores)),
                       "coverage_fraction": min(1.0, covered / (end - start)),
                       "needs_review": (scores[order[0]] - scores[order[1]] < 0.15
                                        or covered / (end - start) < 0.98)})
    return result

def load_models(repo, device):
    import numpy as np
    import scipy
    scipy.inf = np.inf  # upstream msaf compatibility
    import torch
    from ema_pytorch import EMA
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from muq import MuQ
    from musicfm.model.musicfm_25hz import MusicFM25Hz
    from models.SongFormer import Model

    model = Model(OmegaConf.load("configs/SongFormer.yaml"))
    ema = EMA(model, include_online_model=False)
    ema.load_state_dict(load_file("ckpts/SongFormer.safetensors"))
    model.load_state_dict(ema.ema_model.state_dict())
    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(device).eval()
    musicfm = MusicFM25Hz(is_flash=False, stat_path="ckpts/MusicFM/msd_stats.json",
                         model_path="ckpts/MusicFM/pretrained_msd.pt").to(device).eval()
    return model.to(device).eval(), muq, musicfm

def predict(audio_path, models, device, window):
    import librosa
    import torch
    model, muq, musicfm = models
    wav, _ = librosa.load(str(audio_path), sr=24000, mono=True)
    if len(wav) <= 2048:
        raise ValueError("Audio too short")
    duration = len(wav) / 24000
    frames = []
    def encode(chunk):
        tensor = torch.as_tensor(chunk, device=device).unsqueeze(0)
        a = muq(tensor, output_hidden_states=True)["hidden_states"][10]
        b = musicfm.get_predictions(tensor)[1][10]
        return b, a
    def stream(chunk, seconds):
        parts = [encode(chunk[i:i + seconds * 24000])
                 for i in range(0, len(chunk), seconds * 24000)
                 if len(chunk[i:i + seconds * 24000]) > 2048]
        if not parts:
            raise ValueError("No valid encoder frames")
        return [torch.cat([p[k] for p in parts], dim=1) for k in range(2)]
    with torch.inference_mode():
        for offset in range(0, len(wav), 420 * 24000):
            chunk = wav[offset:offset + 420 * 24000]
            if len(chunk) <= 2048:
                break
            print(f"  window {offset / 24000:.1f}s; encoder {window}s", flush=True)
            short = stream(chunk, 30)
            long = stream(chunk, window)
            streams = short + long  # MusicFM30, MuQ30, MusicFMlong, MuQlong
            sizes = [s.shape[1] for s in streams]
            if max(sizes) - min(sizes) > 4:
                raise ValueError(f"Encoder lengths differ: {sizes}")
            count = min(sizes) // 3 * 3
            if count < 3:
                raise ValueError("Insufficient aligned encoder frames")
            embeddings = torch.cat([s[:, :count] for s in streams], dim=-1)
            # Same forward path as infer, without SongFormer boundary postprocessing.
            logits = model.forward_func({"input_embeddings": embeddings,
                        "dataset_ids": torch.tensor([5], dtype=torch.long, device=device),
                        "masks": None})["function_logits"][0]
            probabilities = logits[:, list(LABELS)].softmax(dim=-1).cpu().tolist()
            start = offset / 24000
            for i, row in enumerate(probabilities):
                left = start + i * 3 / 25
                right = min(start + (i + 1) * 3 / 25, duration)
                if right > left:
                    frames.append((left, right, row))
    return frames

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=DEVICE)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    inputs = sorted(APPLE_PATH.rglob("*.json"))
    if args.limit:
        inputs = inputs[:args.limit]
    if not inputs:
        parser.error(f"No Apple JSON found in {APPLE_PATH}")
    pending = [p for p in inputs if not (OUTPUT_PATH / p.relative_to(APPLE_PATH)).exists()]
    if not pending:
        print("Semua output label sudah tersedia. Tidak menimpa koreksi manual.")
        return 0
    repo = SONGFORMER_PATH.expanduser().resolve()
    if not (repo / "src/SongFormer/configs/SongFormer.yaml").exists():
        parser.error("SongFormer source missing. Follow SONGFORMER_SETUP.md.")
    sys.path[:0] = [str(repo / "src/SongFormer"), str(repo / "src/third_party")]
    original = Path.cwd()
    failures = 0
    try:
        os.chdir(repo / "src/SongFormer")
        import torch
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        print(f"Loading pretrained models on {device}...", flush=True)
        models = load_models(repo, device)
        try:
            revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = "unknown"
        for path in pending:
            print(f"Labeling: {path.name}", flush=True)
            try:
                raw_bytes = path.read_bytes()
                document = json.loads(raw_bytes)
                audio = Path(document["audio_path"])
                if not audio.is_file():
                    raise ValueError(f"Audio unavailable on this computer: {audio}")
                frames = predict(audio, models, device, ENCODER_WINDOW_SECONDS)
                labeled = aggregate(document["sections"], frames)
                output = {**document, "sections": labeled, "boundary_locked": True,
                          "label_source": "ASLP-lab/SongFormer", "songformer_revision": revision,
                          "encoder_window_seconds": ENCODER_WINDOW_SECONDS,
                          "apple_json_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                          "score_note": "Uncalibrated mean softmax scores; not guaranteed accuracy."}
                dest = OUTPUT_PATH / path.relative_to(APPLE_PATH)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("x", encoding="utf-8") as handle:
                    json.dump(output, handle, ensure_ascii=False, indent=2, allow_nan=False)
                print(f"OK: {len(labeled)} labeled sections -> {dest}", flush=True)
            except Exception as exc:
                failures += 1
                print(f"ERROR: {path.name}: {exc}", flush=True)
    finally:
        os.chdir(original)
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
