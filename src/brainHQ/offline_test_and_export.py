#!/usr/bin/env python3
"""


Outputs:
 - mf_templates_csv/ (or folder you provide)
   - detections.csv
   - core_corr.png
   - template_core_int16.c
   - template_full_int16.c
   - template_core_resampled.wav
   - template_full_resampled.wav
   - long_resampled.wav
"""

import os, json, math
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import fftconvolve, convolve
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import itertools

EPS = 1e-12


def resample_to_target(sig, sr_from, sr_target):
    if abs(sr_from - sr_target) < 1e-6:
        return sig
    return librosa.resample(sig.astype(float), orig_sr=int(round(sr_from)), target_sr=int(round(sr_target)))

def normxcorr_valid(x, tpl):
    Lt = tpl.shape[0]; Lx = x.shape[0]
    if Lt == 0 or Lt > Lx: return np.array([])
    num = fftconvolve(x, tpl[::-1], mode='same')
    win = np.ones(Lt)
    local_energy = np.sqrt(np.maximum(fftconvolve(x*x, win, mode='same'), 0.0)) + EPS
    tpl_norm = np.linalg.norm(tpl) + EPS
    corr = num / (local_energy * tpl_norm)

    # plt.figure(figsize=(10,3))
    # plt.plot(x, label='bin')
    # plt.plot(tpl, label='tpl')
    # plt.legend()
    # plt.show()

    return np.clip(corr, -1.0, 1.0)

def extract_onset_centered(y, sr, tpl_len_samples, align_env_hop_ms=3):
    hop = int(max(1, round(sr * align_env_hop_ms/1000.0)))
    wlen = max(1, hop*4)
    env = np.sqrt(np.convolve(y*y, np.ones(wlen)/wlen, mode='same'))
    peak = int(np.argmax(env))
    start = max(0, peak - tpl_len_samples//2)
    end = start + tpl_len_samples
    if end > len(y):
        seg = np.zeros(tpl_len_samples)
        avail = max(0, len(y) - start)
        if avail>0:
            seg[:avail] = y[start:start+avail]
        return seg
    return y[start:end]


def to_int16_template(tpl, bits=14):
    if np.max(np.abs(tpl)) < EPS:
        return np.zeros_like(tpl, dtype=np.int16)
    tpl = tpl / (np.max(np.abs(tpl)) + EPS)
    scaled = (tpl * 2**(bits-2)) 
    return scaled.astype(np.int16)

def export_c_array_int16(name, arr):
    s = f"const int16_t {name}[] = {{\n"
    perline = 12
    for i,v in enumerate(arr):
        if (i % perline) == 0:
            s += "  "
        s += f"{int(v)}, "
        if (i % perline) == perline-1:
            s += "\n"
    if (len(arr) % perline) != 0:
        s += "\n"
    s += "};\n"
    s += f"const size_t {name}_len = {len(arr)};\n"
    return s

def export_c_array_float(name, arr):
    s = f"const float {name}[] = {{\n"
    perline = 8
    for i, v in enumerate(arr):
        if (i % perline) == 0:
            s += "  "
        s += f"{float(v):.8g}f, "
        if (i % perline) == perline - 1:
            s += "\n"
    if (len(arr) % perline) != 0:
        s += "\n"
    s += "};\n"
    s += f"const size_t {name}_len = {len(arr)};\n"
    return s

def next_power_of_two(x):
    return 1 << (x - 1).bit_length()

def main(core_ms=75, target_sr=4000, bin_ms=200, bin_overlap=0.5):
    csv = "brainHQ/exampleData.csv"
    time_col = None
    volt_col = None
    template_path = "brainHQ/wav_files"
    out = "templates/INOtemplates_csv"
    core_thr = 0.60
    full_thr = 0.85

    os.makedirs(out, exist_ok=True)

    # Load CSV time & voltage
    df = pd.read_csv(csv)
    t = df.iloc[:,0].apply(lambda x: str(x).split(' ')[0]).astype(float).to_numpy() if time_col is None else df[time_col].to_numpy(dtype=float)
    v = df.iloc[:,1].to_numpy(dtype=float) if volt_col is None else df[volt_col].to_numpy(dtype=float)

    # infer sampling rate
    dt = 0.0001
    sr_inferred = 1.0 / dt
    print(f"Inferred raw SR from CSV time column: {sr_inferred:.2f} Hz")

    # resample long voltage to uniform target_sr if needed
    # first ensure the voltage is uniformly sampled at sr_inferred; if not, interpolate onto uniform grid
    dt = np.diff(t)
    rel_jitter = np.std(dt) / (np.median(dt) + EPS)
    if rel_jitter > 1e-3:
        print(f"Time column has jitter {rel_jitter:.3g}; interpolating to uniform grid at inferred SR {sr_inferred:.2f}")
        t_uniform = np.arange(t[0], t[-1], 1.0/sr_inferred)
        v_uniform = np.interp(t_uniform, t, v)
    else:
        v_uniform = v
        # if time doesn't start at 0, we can still resample based on inferred sr
    # now resample to target_sr
    longy = resample_to_target(v_uniform, sr_inferred, target_sr)
    # bandpass filter at 80-2000 Hz
    b, a = butter(4, [80/(0.5*target_sr), 1500/(0.5*target_sr)], btype='bandpass')
    longy = filtfilt(b, a, longy)
    longy = longy - np.mean(longy)
    
    # iterate through template files in template_path
    tpl_raw = []
    tpl_sr0 = []
    for f in os.listdir(template_path):
        if not f.lower().endswith('.wav'):
            continue
        f = os.path.join(template_path, f)
        print("Using template file:", f)
        tpl, sr0 = sf.read(f)
        if tpl.ndim>1:
            tpl = tpl[:,0]
        tpl_raw.append(tpl)
        tpl_sr0.append(sr0)

    core_samples = int(round(core_ms * target_sr / 1000.0))
    full_samples = int(round(300 * target_sr / 1000.0))
    core_tpls = []
    full_tpls = []
    for i in range(len(tpl_raw)):
        tpl_res = librosa.resample(tpl_raw[i].astype(float), orig_sr=tpl_sr0[i], target_sr=target_sr)
        core_tpl = extract_onset_centered(tpl_res, target_sr, core_samples)
        full_tpl = extract_onset_centered(tpl_res, target_sr, full_samples)
        full_tpl = full_tpl - np.mean(full_tpl)
        core_tpl = core_tpl - np.mean(core_tpl)
        core_tpls.append(core_tpl)
        full_tpls.append(full_tpl)

    plt.figure(figsize=(10,3))
    for i, core_tpl in enumerate(core_tpls):
        t_axis = np.arange(len(core_tpl)) / float(target_sr)
        plt.plot(t_axis, core_tpl, label=f'Template {i+1}')
    plt.legend()
    core_tpl = core_tpls[0]
    # compute core corr trace across long recording
    binlen = int(round(bin_ms * target_sr)/1000)
    # rolling window with 50% overlap
    nbin = int(math.ceil((len(longy) - binlen) / (bin_overlap * binlen))) + 1
    core_corr = np.zeros(nbin)
    t = []
    for b in range(nbin):
        start = int(b * bin_overlap * binlen)
        end = start + binlen
        if start >= len(longy):
            break
        if end > len(longy):
            end = len(longy)
        seg = longy[start:end]
        if len(seg) < core_samples:
            continue
        corr = normxcorr_valid(seg, core_tpl)
        if len(corr) == 0:
            continue
        core_corr[b] = np.abs(corr).max()
        t.append(end/target_sr)


    # find candidate indices where core_corr >= core_thr
    cand_idx = np.where(core_corr >= core_thr)[0]
    score = core_corr[cand_idx]
    print(f"Found {len(cand_idx)} core candidates (thr={core_thr})")
    # print time of candidates
    cand_idx = (cand_idx * bin_overlap * binlen).astype(int)
    print("Candidate times (s):", np.round(cand_idx / target_sr, 3))

    # confirm candidates with full template
    detections = []
    used = set()
    min_sep = int(0.15 * target_sr)
    i=0
    for k in cand_idx:
        if any(abs(k - u) < min_sep for u in used):
            continue
        start_full = int(k)
        end_full = start_full + full_samples
        if end_full > len(longy):
            snippet = np.zeros(full_samples)
            avail = max(0, len(longy) - start_full)
            if avail == 0: continue
            snippet[:avail] = longy[start_full:start_full+avail]
        else:
            snippet = longy[start_full:end_full]
        num = float(np.dot(snippet, full_tpl[::-1]))
        denom = (np.linalg.norm(snippet) * np.linalg.norm(full_tpl) + EPS)
        sc = num / denom
        detections.append({"time_s": float(start_full/target_sr), "sample": int(start_full), "score": float(score[i]), "score_full": float(sc)})
        used.add(start_full)
        i += 1

    print(f"Confirmed {len(detections)} detections (full_thr={full_thr})")
    # save outputs
    if len(detections)>0:
        pd.DataFrame(detections).to_csv(os.path.join(out, "detections.csv"), index=False)
    # plot core corr
    t_axis = np.arange(len(core_corr)) / float(target_sr)
    plt.figure(figsize=(10,3))
    plt.plot(t, core_corr, label='core_corr')
    for d in detections:
        plt.axvline(d['time_s'], color='k', linestyle='--')
        plt.text(d['time_s'], 0.9, f"{d['score']:.2f}", rotation=90)
    plt.axhline(core_thr, color='r', linestyle=':', label='core_thr')
    plt.xlabel("Time (s)"); plt.ylabel("core corr"); plt.title("Core corr trace")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "core_corr.png"), dpi=150)
    plt.close()

    sf.write(os.path.join(out, "template_core_resampled.wav"), core_tpl, target_sr)
    sf.write(os.path.join(out, "template_full_resampled.wav"), full_tpl, target_sr)
    sf.write(os.path.join(out, "long_resampled.wav"), longy, target_sr)

    # save templates and mock data as C arrays
    template_int14 = to_int16_template(core_tpl, bits=14)
    with open(os.path.join(out, "template_int16.h"), "w") as fh:
        fh.write(export_c_array_int16("template_int16", template_int14))

    summary = {"target_sr": target_sr, "core_samples": core_samples, "full_samples": full_samples,
               "core_thr": core_thr, "full_thr": full_thr, "n_detections": len(detections)}
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("Outputs saved to", out)
    if len(detections) == 0:
        print("No detections found; consider lowering thresholds or verifying time/voltage columns.")

    # want to maximize margin between detected peaks and non-detected peaks
    if len(detections) != 3:
        return 0
    det_scores = np.array([d['score'] for d in detections])
    non_det_scores = core_corr[core_corr < core_thr]
    if len(non_det_scores) == 0:
        margin = float(np.min(det_scores) - core_thr)
    else:
        margin = float(np.mean(det_scores) - np.max(non_det_scores))

    return margin


if __name__ == "__main__":
    core_ms_values = [75]
    target_sr_values =[4000]
    bin_ms_values = [200]
    bin_overlap = [0.5]

    best_score = -1
    best_params = None

    for core_ms, target_sr, bin_ms, bin_overlap in itertools.product(core_ms_values, target_sr_values, bin_ms_values, bin_overlap):
        # Call your main detection function here, passing core_ms and target_sr
        # For example, you might refactor your main() to accept these as arguments
        score = main(core_ms=core_ms, target_sr=target_sr, bin_ms=bin_ms, bin_overlap=bin_overlap)  # implement this
        print(f"core_ms={core_ms}, target_sr={target_sr}, bin_ms={bin_ms}, bin_overlap={bin_overlap}, score={score}")
        if score > best_score or (score == best_score and (core_ms < best_params[0] or target_sr < best_params[1])):
            best_score = score
            best_params = (core_ms, target_sr, bin_ms, bin_overlap)

    print("Best parameters:", best_params, "with score:", best_score)


