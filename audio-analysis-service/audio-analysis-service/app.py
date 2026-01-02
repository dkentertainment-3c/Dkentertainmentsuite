from flask import Flask, request, jsonify
import librosa
import numpy as np
import io

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"ok": True})

def pick_rhythmic_window(y, sr, window_sec=35):
    """Pick the most rhythmic window to avoid quiet intros/outros."""
    if len(y) <= sr * window_sec:
        return y, {"start_sec": 0, "end_sec": round(len(y) / sr, 2)}

    hop = sr * 5  # slide every 5 seconds
    win = sr * window_sec
    best_score = -1
    best_start = 0

    for start in range(0, len(y) - win, hop):
        seg = y[start:start + win]
        onset = librosa.onset.onset_strength(y=seg, sr=sr)
        score = float(np.mean(onset))  # simple beat salience proxy
        if score > best_score:
            best_score = score
            best_start = start

    best_end = min(best_start + win, len(y))
    return y[best_start:best_end], {
        "start_sec": round(best_start / sr, 2),
        "end_sec": round(best_end / sr, 2),
    }

def bpm_candidates(y, sr):
    """Return BPM top candidates + half/double options."""
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    # per-frame tempo estimates (gives a distribution)
    tempi = librosa.feature.tempo(onset_envelope=onset, sr=sr, aggregate=None)

    if tempi is None or len(tempi) == 0:
        return None, [], 0.0, True

    # Most common tempos
    tempi = tempi[np.isfinite(tempi)]
    if len(tempi) == 0:
        return None, [], 0.0, True

    # Histogram vote
    hist, edges = np.histogram(tempi, bins=60, range=(40, 220))
    top_bins = np.argsort(hist)[::-1][:3]
    cands = []
    for b in top_bins:
        bpm = float((edges[b] + edges[b+1]) / 2)
        cands.append(bpm)

    bpm = cands[0] if cands else float(np.median(tempi))

    # Half/double ambiguity check
    ambiguous = False
    if len(cands) >= 2:
        t1, t2 = cands[0], cands[1]
        if abs(t1 - 2*t2) < 2 or abs(2*t1 - t2) < 2:
            ambiguous = True

    # Confidence: ratio of top bin vs total
    conf = float(hist[top_bins[0]] / max(1, np.sum(hist)))
    conf = min(max(conf, 0.0), 1.0)

    # Add explicit half/double candidates (if in sane range)
    extra = []
    if bpm and 40 <= bpm <= 220:
        if bpm/2 >= 40: extra.append(round(bpm/2, 2))
        if bpm*2 <= 220: extra.append(round(bpm*2, 2))

    full_candidates = []
    for x in (cands + extra):
        if x not in full_candidates:
            full_candidates.append(x)

    return bpm, full_candidates[:5], conf, ambiguous

def key_candidates(y, sr):
    """
    Simple chroma-based key guess.
    (This is a solid upgrade over 'AI transcription', and the UI will allow quick correction.)
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    keys = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

    # Basic major/minor templates (Krumhansl)
    major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

    def score_for(profile):
        profile = (profile - profile.mean()) / (profile.std() + 1e-9)
        v = (chroma_mean - chroma_mean.mean()) / (chroma_mean.std() + 1e-9)
        scores = [float(np.dot(np.roll(profile, i), v)) for i in range(12)]
        return np.array(scores)

    maj = score_for(major_profile)
    minr = score_for(minor_profile)

    maj_best = int(np.argmax(maj))
    min_best = int(np.argmax(minr))

    candidates = []
    # top 3 major
    for i in np.argsort(maj)[::-1][:2]:
        candidates.append({"key": f"{keys[int(i)]} major", "score": float(maj[int(i)])})
    # top 3 minor
    for i in np.argsort(minr)[::-1][:2]:
        candidates.append({"key": f"{keys[int(i)]} minor", "score": float(minr[int(i)])})

    # pick best overall
    if maj[maj_best] >= minr[min_best]:
        best = f"{keys[maj_best]} major"
        conf = float((maj[maj_best] - np.median(maj)) / (np.std(maj) + 1e-9))
    else:
        best = f"{keys[min_best]} minor"
        conf = float((minr[min_best] - np.median(minr)) / (np.std(minr) + 1e-9))

    # normalize confidence to 0..1-ish
    conf = max(0.0, min(1.0, conf / 3.0))

    # sort candidates best-first
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:3]
    ambiguous = conf < 0.35

    return best, candidates, conf, ambiguous

@app.post("/analyze")
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "missing file field named 'file'"}), 400

    audio_bytes = request.files["file"].read()
    if not audio_bytes:
        return jsonify({"error": "empty file"}), 400

    # Load audio
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True)

    # Choose most rhythmic segment
    y_seg, window = pick_rhythmic_window(y, sr, window_sec=35)

    bpm, bpm_cands, bpm_conf, bpm_amb = bpm_candidates(y_seg, sr)
    key, key_cands, key_conf, key_amb = key_candidates(y_seg, sr)

    return jsonify({
        "bpm": bpm,
        "bpm_candidates": bpm_cands,
        "bpm_confidence": bpm_conf,
        "key": key,
        "key_candidates": key_cands,
        "key_confidence": key_conf,
        "analyzed_window": window,
        "flags": {
            "half_double_ambiguous": bpm_amb,
            "key_ambiguous": key_amb
        }
    })
