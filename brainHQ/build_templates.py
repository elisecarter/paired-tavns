# build_templates_and_evaluate_simple.py
# Offline template builder + evaluator for stereotyped UI sounds.
# Usage:
# 1) Put labeled wavs into ./wav_files/
#    Filenames should start with the label and an underscore, e.g.:
#      correctA_001.wav, correctB_002.wav, wrong1_003.wav
# 2) Run this script: python build_templates_and_evaluate_simple.py
#
# Outputs saved to ./mf_templates/:
#  - templates.npz         (SR, tmpl_len, template_struct)
#  - corr_scores_per_file.csv
#  - PR_<LABEL>.png for each label
#  - overlay_envelopes.png
#
# Tune parameters below if necessary.

import os
import scipy
import glob
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------- PARAMETERS (tune if needed) ----------
# Resolve paths relative to this script's directory
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WAV_DIR = os.path.join(_BASE_DIR, "wav_files")
OUT_DIR = os.path.join(_BASE_DIR, "templates")
tmpl_len_ms = 100         # initial template length in ms (adjust after you inspect envelopes)
align_env_hop_ms = 3      # hop used for onset detection (ms)
proto_count = 3           # keep up to this many prototypes per label
min_examples_for_proto = 2# require at least this many examples to make prototypes
PR_STEPS = 200            # resolution for PR sweep
EPS = 1e-12
# ----------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

def load_mono(path):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return y.astype(float), sr

def rms_envelope(y, win_samples):
    """Compute moving RMS via cumulative sums (no SciPy), return 'same' length.
    Uses trailing window averaging padded to same length by edge values.
    """
    if win_samples < 1:
        win_samples = 1
    y = np.asarray(y, dtype=float).ravel()
    sq = y * y
    # cumulative sum trick for moving sum over window
    c = np.cumsum(np.concatenate(([0.0], sq)))
    sums_valid = c[win_samples:] - c[:-win_samples]
    rms_valid = np.sqrt(np.maximum(sums_valid / float(win_samples), 0.0))
    # pad to 'same' length, center the window approximately
    pad_left = win_samples // 2
    pad_right = len(y) - (len(rms_valid) + pad_left)
    if pad_right < 0:
        pad_right = 0
    rms_same = np.pad(rms_valid, (pad_left, pad_right), mode='edge')[: len(y)]
    return rms_same

def _fft_convolve_valid(x, k):
    """Valid-length convolution via FFT using NumPy only.
    Returns conv(x, k) with mode='valid'.
    """
    x = np.asarray(x, dtype=float).ravel()
    k = np.asarray(k, dtype=float).ravel()
    n = x.size
    m = k.size
    if m > n:
        return np.array([])
    L = n + m - 1
    N = 1 << (L - 1).bit_length()  # next pow2
    X = np.fft.rfft(x, N)
    K = np.fft.rfft(k, N)
    y = np.fft.irfft(X * K, N)[:L]
    start = m - 1
    stop = start + (n - m + 1)
    return y[start:stop]

def normxcorr_1d(tpl, x):
    """Normalized cross-correlation ('valid') using NumPy only.
    corr[i] = sum_j x[i+j]*tpl[j] / (||tpl|| * ||x[i:i+Lt]||)
    """
    tpl = np.asarray(tpl, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    Lt = tpl.size
    Lx = x.size
    if Lt == 0 or Lt > Lx:
        return np.array([])
    # numerator: conv(x, reverse(tpl), valid)
    # num = _fft_convolve_valid(x, tpl[::-1])
    num = scipy.signal.fftconvolve(x, tpl[::-1], mode='valid')  # alternative using SciPy
    # denominator: local window energy of x times norm(tpl)
    sq = x * x
    c = np.cumsum(np.concatenate(([0.0], sq)))
    win_sums = c[Lt:] - c[:-Lt]
    local_energy = np.sqrt(np.maximum(win_sums, 0.0)) + EPS
    tpl_norm = np.linalg.norm(tpl) + EPS
    denom = local_energy * tpl_norm
    corr = num / denom
    return np.clip(corr, -1.0, 1.0)

# ---------- load files ----------
files = sorted(glob.glob(os.path.join(WAV_DIR, "*.wav")))
if len(files) == 0:
    raise SystemExit(f"No wav files found in {WAV_DIR}. Create {WAV_DIR}/ and add labeled wavs.")

nFiles = len(files)
labels = []
audio_raw = []
srs = []
filenames = []
for fn in files:
    y, sr = load_mono(fn)
    audio_raw.append(y)
    srs.append(sr)
    filenames.append(os.path.basename(fn))
    label = os.path.splitext(os.path.basename(fn))[0]
    labels.append(label)

unique_labels = sorted(list(set(labels)))
print(f"Found {nFiles} files with {len(unique_labels)} unique labels: {', '.join(unique_labels)}")

# ---------- choose SR (use max to minimize upsampling issues) ----------
SR = int(8000)
print(f"Using SR = {SR}... resampling.")
for i in range(nFiles):
    if srs[i] != SR:
        num = int(round(len(audio_raw[i]) * SR / srs[i]))
        audio_raw[i] = scipy.signal.resample(audio_raw[i], num)


# ---------- resample & onset alignment ----------
align_hop = max(1, int(round(SR * (align_env_hop_ms / 1000.0))))  # samples
y_rs = []
onsets = []
env_collection = []

for i in range(nFiles):
    y = audio_raw[i]
    sr = srs[i]
    y = y.ravel()
    y_rs.append(y)
    # compute short-window RMS envelope for onset detection
    wlen = max(1, align_hop * 4)
    env = rms_envelope(y, wlen)
    # onset estimate = argmax of envelope
    idx = int(np.argmax(env)) if env.size > 0 else 0
    onsets.append(max(0, idx))
    # downsample envelope for plotting (uniform time vector)
    # build time vector for env same length as y: env is same length due to 'same' conv
    env_collection.append(env)

# ---------- template extraction ----------
tmpl_len = int(round(SR * tmpl_len_ms / 1000.0))
print(f"Template length = {tmpl_len} samples ({tmpl_len_ms:.1f} ms)")

examples_by_label = {lab: [] for lab in unique_labels}

for i in range(nFiles):
    lab = labels[i]
    y = y_rs[i]
    t0 = onsets[i]
    # put onset ~50% into template (start = t0 - tmpl_len/2)
    start_idx = max(0, int(t0 - round(tmpl_len/2)))
    stop_idx = start_idx + tmpl_len
    seg = np.zeros(tmpl_len, dtype=float)
    if stop_idx <= len(y):
        seg[:] = y[start_idx:stop_idx]
    else:
        avail = max(0, len(y) - start_idx)
        if avail > 0:
            seg[:avail] = y[start_idx:start_idx+avail]
        # remainder stays zero
    seg = seg - np.mean(seg)
    examples_by_label[lab].append(seg)

# compute mean template and prototypes per label
template_struct = {}
for lab in unique_labels:
    exs = examples_by_label.get(lab, [])
    if len(exs) == 0:
        raise RuntimeError(f"No examples found for label {lab} (unexpected).")
    A = np.vstack([e.reshape(1, -1) for e in exs])
    meanTpl = np.mean(A, axis=0).reshape(-1)
    mean_norm = np.linalg.norm(meanTpl) + EPS
    meanTpl = meanTpl / mean_norm
    # choose prototypes closest to mean
    dists = np.sqrt(np.sum((A - meanTpl.reshape(1,-1))**2, axis=1))
    idxs = np.argsort(dists)
    np_count = min(proto_count, A.shape[0]) if A.shape[0] >= min_examples_for_proto else A.shape[0]
    protos = []
    for p in range(np_count):
        v = A[idxs[p], :].reshape(-1)
        v = v / (np.linalg.norm(v) + EPS)
        protos.append(v)
    template_struct[lab] = {"mean": meanTpl, "prototypes": protos, "n_examples": A.shape[0]}

# save templates (npz) and plot mean+protos for visual check
np.savez(os.path.join(OUT_DIR, "templates.npz"), SR=SR, tmpl_len=tmpl_len, template_struct=template_struct)
print(f"Saved templates to {os.path.join(OUT_DIR, 'templates.npz')}")
plt.figure(figsize=(8,4))
colors = plt.cm.tab10.colors
for li, lab in enumerate(unique_labels):
    tpl_info = template_struct[lab]
    meanTpl = tpl_info["mean"]
    time = np.arange(meanTpl.size) / SR * 1000  # ms
    plt.subplot(5,1,li+1)
    plt.plot(time, meanTpl, linewidth=2.0, color=colors[li % len(colors)], label=f"{lab} mean")
    plt.ylabel(lab)
plt.xlabel('time (ms)')
tpl_png = os.path.join(OUT_DIR, "templates.png") 
plt.savefig(tpl_png, dpi=150)
plt.close()
print(f"Saved template plot to {tpl_png}")

# ---------- compute normalized cross-correlation scores per file vs each label ----------
labelNames = list(template_struct.keys())
corr_scores = np.zeros((nFiles, len(labelNames)), dtype=float)

for i in range(nFiles):
    x = y_rs[i]
    for li, lab in enumerate(labelNames):
        tpls = [template_struct[lab]["mean"]] + template_struct[lab]["prototypes"]
        maxval = -np.inf
        for tpl in tpls:
            c = normxcorr_1d(tpl, x)
            if c.size > 0:
                m = float(np.max(c))
                if m > maxval:
                    maxval = m
        if np.isinf(maxval):
            maxval = 0.0
        corr_scores[i, li] = maxval

# write CSV
csvfile = os.path.join(OUT_DIR, "corr_scores_per_file.csv")
with open(csvfile, "w") as f:
    header = "filename," + ",".join(labelNames) + ",true_label\n"
    f.write(header)
    for i in range(nFiles):
        row = filenames[i] + "," + ",".join([f"{corr_scores[i,li]:.6f}" for li in range(len(labelNames))]) + f",{labels[i]}\n"
        f.write(row)
print(f"Saved per-file corr scores to {csvfile}")

# ---------- overlay envelopes (normalized) for visual inspection ----------
# normalize and align envelopes to onset (we used downsampled envs) and plot mean per label
env_matrix = []
maxlen = max([len(e) for e in env_collection])
# pad with NaN
env_matrix = np.full((nFiles, maxlen), np.nan, dtype=float)
for i, e in enumerate(env_collection):
    env_norm = (e - np.min(e)) / (np.max(e) - np.min(e) + EPS)
    env_matrix[i, :len(env_norm)] = env_norm

plt.figure(figsize=(8,4))
colors = plt.cm.tab10.colors
for li, lab in enumerate(unique_labels):
    idxs = [i for i,l in enumerate(labels) if l==lab]
    if len(idxs) == 0:
        continue
    mean_env = np.nanmean(env_matrix[idxs, :], axis=0)
    y = y_rs[li]
    t_env = np.linspace(0, len(y)/SR, num=env.size)
    plt.plot(np.arange(mean_env.size)/SR*1000, mean_env, linewidth=1.5, color=colors[li % len(colors)], label=lab)
plt.legend(loc='best')
plt.xlabel('time (ms)')
plt.ylabel('Normalized envelope (mean)')
plt.title('Average normalized envelopes by label (quick visual check)')
plt.tight_layout()
overlay_png = os.path.join(OUT_DIR, "overlay_envelopes.png")
plt.savefig(overlay_png, dpi=150)
plt.close()
print(f"Saved envelope overlay to {overlay_png}")

# ---------- Precision-Recall (PR) and suggested thresholds ----------
for li, lab in enumerate(labelNames):
    ytrue = np.array([1 if lab == lbl else 0 for lbl in labels], dtype=int)
    scores = corr_scores[:, li]
    thr_list = np.linspace(0.0, 1.0, PR_STEPS)
    prec = np.zeros_like(thr_list)
    rec = np.zeros_like(thr_list)
    for k, thr in enumerate(thr_list):
        preds = scores >= thr
        tp = int(np.sum(preds & (ytrue==1)))
        fp = int(np.sum(preds & (ytrue==0)))
        fn = int(np.sum((~preds) & (ytrue==1)))
        prec[k] = tp / max(1, (tp+fp))
        rec[k] = tp / max(1, (tp+fn))
    # pick threshold where precision >= 0.98 if exists
    candidates = np.where(prec >= 0.98)[0]
    if candidates.size > 0:
        chosen_idx = candidates[0]
        chosen_thr = float(thr_list[chosen_idx])
    else:
        # fallback: choose thr at max F1
        f1 = 2.0 * (prec * rec) / (prec + rec + EPS)
        bi = int(np.nanargmax(f1))
        chosen_thr = float(thr_list[bi])
        chosen_idx = bi
    # save PR plot
    plt.figure(figsize=(5,4))
    plt.plot(rec, prec, '-o', markersize=4)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR curve: {lab}')
    # mark chosen point
    plt.plot(rec[chosen_idx], prec[chosen_idx], 'ro', label=f"thr={chosen_thr:.3f}")
    plt.legend()
    pr_png = os.path.join(OUT_DIR, f"PR_{lab}.png")
    plt.tight_layout()
    plt.savefig(pr_png, dpi=150)
    plt.close()
    print(f"Label {lab}: suggested threshold = {chosen_thr:.3f} (plot saved to {pr_png})")

print("Done. Check folder", OUT_DIR, "for templates, CSV, and plots.")