import os
import pandas as pd
import json
import numpy as np
from matplotlib import pyplot as plt
import datetime
import argparse
import sys
import traceback


def _compute_trace_summary(time, values):
    t = np.asarray(time, dtype=float)
    v = np.asarray(values, dtype=float)
    valid_mask = np.isfinite(t) & np.isfinite(v)
    t = t[valid_mask]
    v = v[valid_mask]
    if t.size == 0 or v.size == 0:
        return {}

    order = np.argsort(t)
    t = t[order]
    v = v[order]

    max_idx = int(np.nanargmax(v))
    min_idx = int(np.nanargmin(v))
    try:
        auc = float(np.trapezoid(v, x=t))
        abs_auc = float(np.trapezoid(np.abs(v), x=t))
    except Exception:
        auc = np.nan
        abs_auc = np.nan

    return {
        'mean': float(np.nanmean(v)),
        'std': float(np.nanstd(v)),
        'median': float(np.nanmedian(v)),
        'iqr': float(np.subtract(*np.percentile(v, [75, 25]))),
        'min': float(np.nanmin(v)),
        'max': float(np.nanmax(v)),
        'time_to_max': float(t[max_idx]),
        'time_to_min': float(t[min_idx]),
        'auc': auc,
        'abs_auc': abs_auc,
    }


def _build_next_response_maps(event_series):
    events = event_series.fillna('').astype(str).str.lower().to_numpy()
    resp_mask = np.char.find(events.astype(str), 'response') >= 0
    corr_mask = resp_mask & (np.char.find(events.astype(str), 'correct') >= 0)

    n = len(events)
    next_resp = np.full(n, -1, dtype=int)
    next_corr = np.full(n, -1, dtype=int)
    last_resp = -1
    last_corr = -1
    for i in range(n - 1, -1, -1):
        if resp_mask[i]:
            last_resp = i
        if corr_mask[i]:
            last_corr = i
        next_resp[i] = last_resp
        next_corr[i] = last_corr
    return next_resp, next_corr


def prepare_block_events(ts_data, experiment_type):
    if 'Event' not in ts_data.columns:
        return pd.DataFrame(columns=['event', 'trial'])

    df_events = pd.DataFrame({'event': ts_data['Event'].dropna().astype(str)})
    if experiment_type in ['SCWT', 'StroopSquared', 'PLRT']:
        df_events = df_events[df_events['event'].str.contains('flash_start|cue|response', case=False, na=False)].copy()
        if df_events.empty:
            return pd.DataFrame(columns=['event', 'trial'])

        event_values = df_events['event'].astype(str).to_numpy()
        for pos in range(len(event_values)):
            current = event_values[pos]
            if 'cue' in current.lower():
                if pos + 1 < len(event_values):
                    response = event_values[pos + 1].split('_')[-1]
                    event_values[pos] = f"{current}_{response}"
                else:
                    event_values[pos] = ''
        df_events['event'] = event_values
        df_events = df_events[df_events['event'].str.len() > 0]
        df_events = df_events[~df_events['event'].str.contains('response_', case=False, na=False)]
        df_events['trial'] = np.arange(1, len(df_events) + 1, dtype=int)

    return df_events

def epoch_by_event(block_data, event, event_df, pre_event_dur: float = 5.0, post_event_dur: float = 5.0, block_cfg=None, baseline_method='delta'):
    """Returns a tidy dataframe with trial-epoch data for the specified event."""
    # Convert timestamps to numeric seconds (handle datetime or numeric)
    if np.issubdtype(block_data['Timestamps'].dtype, np.datetime64) or hasattr(block_data['Timestamps'].dtype, 'tz'):
        times = pd.to_datetime(block_data['Timestamps']).astype('datetime64[ns]').astype('int64') / 1e9
    else:
        times = block_data['Timestamps'].to_numpy(dtype=float)

    # Compute dt (seconds) robustly
    diffs = np.diff(times)
    if len(diffs) == 0:
        return None, None
    dt = float(np.median(diffs))
    if dt <= 0 or np.isnan(dt):
        return None, None

    pre_samples = max(1, int(np.round(pre_event_dur / dt)))
    post_samples = max(1, int(np.round(post_event_dur / dt)))

    timeseries_data = []
    feature_rows = []
    skipped_missing_trials = 0
    event_positions = block_data.index.get_indexer(event_df.index)
    valid_event_mask = event_positions >= 0
    event_positions = event_positions[valid_event_mask]
    if event_positions.size == 0:
        return None, None

    trial_values = event_df['trial'].to_numpy() if 'trial' in event_df.columns else np.full(len(event_df), np.nan)
    trial_values = trial_values[valid_event_mask]

    next_resp = next_corr = None
    if 'Event' in block_data.columns:
        next_resp, next_corr = _build_next_response_maps(block_data['Event'])

    subj_id = block_cfg.get('participant_ID', block_cfg.get('ID')) if block_cfg else None
    order_val = block_cfg.get('order') if block_cfg else None
    datetime_val = block_cfg.get('datetime') if block_cfg else None
    condition_val = block_cfg.get('condition') if block_cfg else None
    experiment_val = block_cfg.get('experiment') if block_cfg else None
    block_val = block_cfg.get('block_no') if block_cfg else None

    for i, pos in enumerate(event_positions):

        start_ind = pos - pre_samples
        end_ind = pos + post_samples
        if start_ind < 0 or end_ind > (len(block_data) - 1):
            continue

        rel_time = times[start_ind:end_ind] - times[pos]

        rt = np.nan
        if next_resp is not None and next_corr is not None and pos + 1 < len(block_data):
            cand_corr = next_corr[pos + 1]
            cand_resp = next_resp[pos + 1]
            resp_pos = cand_corr if cand_corr != -1 else cand_resp
            if resp_pos != -1:
                rt = float(times[resp_pos] - times[pos])

        for sig in block_data.columns.difference(['Timestamps', 'Event', 'nSeq']):
            values = block_data[sig].iloc[start_ind:end_ind].to_numpy(dtype=float)
            
            # outlier_threshold = 3 * np.nanstd(values)
            # outliers = np.abs(values - np.nanmean(values)) > outlier_threshold
            # values[outliers] = np.nan  # mark outliers as NaN
            # if sig == 'pupilDiameter':
            #     plt.figure(figsize=(10, 4))
            #     plt.plot(rel_time, values - np.nanmean(values), label='Original')
            #     plt.axhline(y=outlier_threshold, color='r', linestyle='--', label='Outlier Threshold')
            #     plt.axhline(y=-outlier_threshold, color='r', linestyle='--')
            #     plt.title(f'percent bad data: {np.sum(np.isnan(values)) / len(values) * 100:.2f}%')
            #     plt.legend()
            #     plt.show()
            
            # determine percent interpolated
            if np.sum(np.isnan(values)) / len(values) > 0.40:
                # skip trials with >40% bad/missing data
                skipped_missing_trials += 1
                continue

            # interpolate missing data linearly
            nans = np.isnan(values)
            if np.any(nans):
                values[nans] = np.interp(rel_time[nans], rel_time[~nans], values[~nans])
             
    
            
            

            # baseline window derived from pre_event_dur
            BL_mask = (rel_time >= -pre_event_dur) & (rel_time < 0)
            BL_vals = values[BL_mask]
            if BL_vals.size == 0:
                BL_mean = 0.0
                BL_std = 1.0
                baseline_flag = False
            else:
                BL_mean = np.nanmean(BL_vals)
                BL_std = np.nanstd(BL_vals)
                baseline_flag = True
                if np.isnan(BL_std) or BL_std == 0:
                    # avoid divide-by-zero; mark invalid baseline
                    BL_std = 1.0
                    baseline_flag = False

            # apply selected baseline correction method
            if baseline_method == 'zscore':
                z_values = (values - BL_mean) / BL_std
            elif baseline_method == 'delta':
                z_values = values - BL_mean
            elif baseline_method == 'percent_change':
                if BL_mean != 0:
                    z_values = ((values - BL_mean) / BL_mean) * 100.0
                else:
                    z_values = values * 0.0  # avoid divide-by-zero
            elif baseline_method == 'none':
                z_values = values # no baseline correction

            # Extract features on post-event window (extract_features expects numpy arrays)
            features = extract_features(z_values, rel_time, sig)
            
            
            
            # if PLRT, caluclate additional features
            if 'PLRT' in (block_cfg.get('experiment', '') if block_cfg else '') and sig == 'pupilDiameter':
                plrt_features = extract_plrt_features(z_values, rel_time)
                if isinstance(features, dict):
                    features.update(plrt_features)
                else:
                    features = plrt_features
            

            n_samples = len(rel_time)
            if n_samples != len(z_values):
                continue

            trial_val = int(trial_values[i]) if np.isfinite(trial_values[i]) else None
            ts_long = pd.DataFrame({
                'id': subj_id,
                'order': order_val,
                'datetime': datetime_val,
                'condition': condition_val,
                'experiment': experiment_val,
                'block': block_val,
                'trial': trial_val,
                'time': np.round(rel_time, 3),
                'signal_type': sig,
                'event': str(event).strip(),
                'value': np.round(z_values, 3),
                'rt': np.round(rt, 3),
            })
            timeseries_data.append(ts_long)

            feature_dict = {
                'id': subj_id,
                'order': order_val,
                'datetime': datetime_val,
                'condition': condition_val,
                'experiment': experiment_val,
                'block': block_val,
                'trial': trial_val,
                'event': str(event).strip(),
                'signal_type': sig,
                'baseline_valid': baseline_flag,
                'baseline_mean': float(BL_mean),
                'baseline_std': float(BL_std)
            }
            if isinstance(features, dict):
                feature_dict.update(features)
            feature_rows.append(feature_dict)

    # Safely concatenate only when we have collected data; otherwise return (None, None)
    timeseries_df = pd.concat(timeseries_data, ignore_index=True) if timeseries_data else None
    feature_df = pd.DataFrame(feature_rows) if feature_rows else None

    if skipped_missing_trials > 0:
        print(f"Skipped {skipped_missing_trials} epochs for event {event} due to excessive missing data (>40%).")

    if timeseries_df is None and feature_df is None:
        return None, None

    return timeseries_df, feature_df

def extract_plrt_features(trial_data, time):
    """Extracts PLRT-specific features from pupil diameter trial data."""
    features = {}
    if trial_data is None:
        return features
    
    dt = float(np.mean(np.diff(time))) if len(time) > 1 else 0.0
    velocity = np.gradient(trial_data, dt) if dt > 0 else np.full(len(trial_data), np.nan)
    const_amplitude = np.nan
    const_latency = np.nan
    
    # maximum constriction velocity: window 0-1s
    constriction_mask = (time >= 0) & (time <= 1)
    const_t = time[constriction_mask]
    if dt > 0 and np.any(constriction_mask):
        mcv = np.min(velocity[constriction_mask])
        mcv_latency = const_t[np.argmin(velocity[constriction_mask])]
    else:
        mcv = np.nan
        mcv_latency = np.nan
    
    # maximum dilation velocity: window relative to mcv
    dilation_mask = (time > 0.5) & (time <= mcv_latency+1.0)
    dil_t = time[dilation_mask]
    if dt > 0 and np.any(dilation_mask):
        mdv = np.max(velocity[dilation_mask])
        mdv_latency = dil_t[np.argmax(velocity[dilation_mask])]
    else:
        mdv = np.nan
        mdv_latency = np.nan
    
    # constriction amplitude/latency (between mcv and mdv)
    if dt > 0 and not np.isnan(mcv_latency) and not np.isnan(mdv_latency):
        if mcv_latency < mdv_latency:
            ca_mask = (time >= mcv_latency) & (time <= mdv_latency)
            if np.any(ca_mask):
                ca_data = trial_data[ca_mask]
                const_amplitude = ca_data.min() - np.nanmin(trial_data)
                const_latency = time[ca_mask][ca_data.argmin()] if len(ca_data) > 0 else np.nan
            else:
                const_amplitude = np.nan
                const_latency = np.nan
        else:
            const_amplitude = np.nan
            const_latency = np.nan
    features.update({
        'max_constriction_velocity': mcv,
        'max_dilation_velocity': mdv,
        'mcv_latency': mcv_latency,
        'mdv_latency': mdv_latency,
        'const_amplitude': const_amplitude,
        'const_latency': const_latency,
    })
    
    # plt.figure(figsize=(10, 5))
    # plt.plot(time, trial_data, label='pupilDiameter')
    # plt.axvline(x=mcv_latency, color='r', linestyle='--', label='Max Constriction Velocity')
    # plt.axvline(x=mdv_latency, color='g', linestyle='--', label='Max Dilation Velocity')
    # plt.axvline(x=const_latency, color='b', linestyle='--', label='Constriction Latency')
    # plt.legend()
    # plt.show()

    return features

def extract_features(trial_data, time, signal, thresh=0):
    """Extracts features from the trial timeseries."""
    if trial_data is None:
        return None
    

    post_event_mask = (time >= 0)
    trial_data = trial_data[post_event_mask]
    time = time[post_event_mask]
    
    features = _compute_trace_summary(time, trial_data)
        
    
        
    if signal == 'SCR':
        signs = np.sign(trial_data - thresh) # threshold at zero
        zcs = np.concatenate(([0], np.diff(signs)))  # prepend 0 to match length

        # find positive zero crossings
        pos_crossings = np.where(zcs > 0)
        neg_crossings = np.where(zcs < 0)

        max_inds = []
        num_peaks = 0
        for p in pos_crossings[0]:
            # if p > neg_crossings[0][-1]:
            #     continue
            # Find next largest index in neg_crossings
            if any(neg_crossings[0] > p):
                n = neg_crossings[0][neg_crossings[0] > p][0]
            else:
                n = len(trial_data)
            # Find max_index in phasic from p -> n
            max_inds.append(np.argmax(trial_data[p:n]) + p)
            num_peaks += 1
        # features['num_peaks'] = num_peaks
        
        
        # plt.figure(figsize=(10, 5))
        # sns.lineplot(x=time, y=trial_data, label=signal)
        # plt.axhline(y=max_value, color='r', linestyle='--', label='Max Peak')
        # plt.axhline(y=min_value, color='g', linestyle='--', label='Min Trough')
        # plt.axvline(x=peak_time, color='r', linestyle=':', label='Peak Time')
        # plt.axvline(x=trough_time, color='g', linestyle=':', label='Trough Time')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Pupil Diameter (z)')
        # plt.legend()
        # plt.show()

    # # Extract features
    # features = {
    #     'mean': trial_data.mean(),
    #     'std': trial_data.std(),
    #     'min': trial_data.min(),
    #     'max': trial_data.max(),
    #     'median': trial_data.median(),
    #     '25%': trial_data.quantile(0.25),
    #     '75%': trial_data.quantile(0.75),
    # }
    

    return features

def compute_timeseries_metrics(g: pd.DataFrame) -> dict:
    """Compute metrics on an average subject/condition trace (columns: time, val)."""
    try:
        t = g['time'].to_numpy(dtype=float)
        v = g['val'].to_numpy(dtype=float)
        return _compute_trace_summary(t, v)
    except Exception:
        return {}


def _add_sem(df, value_col):
    out = df.copy()
    out['sem'] = out['std'] / np.sqrt(out['n_subjects'].clip(lower=1))
    out['sem'] = out['sem'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _plot_condition_mean(subject_mean, condition_mean, condition_colors, y_label, title, output_path, rt_means=None):
    plt.figure(figsize=(5.5, 4.25))
    plt.axvline(0, color='k', linestyle='--')

    for cond, cond_df in condition_mean.groupby('condition'):
        col = condition_colors.get(str(cond), None)
        if col is not None:
            plt.fill_between(cond_df['time'], cond_df['mean'] - cond_df['sem'], cond_df['mean'] + cond_df['sem'], linewidth=0, color=col, alpha=0.3, label=f"{cond} (sem)")
            plt.plot(cond_df['time'], cond_df['mean'], color=col, label=f"{cond} (mean)")
        else:
            plt.fill_between(cond_df['time'], cond_df['mean'] - cond_df['sem'], cond_df['mean'] + cond_df['sem'], linewidth=0, alpha=0.3, label=f"{cond} (sem)")
            plt.plot(cond_df['time'], cond_df['mean'], label=f"{cond} (mean)")

    if rt_means is not None:
        for cond, rt_val in rt_means.items():
            col = condition_colors.get(str(cond), None)
            plt.axvline(float(rt_val), color=col if col is not None else 'gray', linestyle=':', linewidth=1.5, label=f"{cond} mean RT")

    if not subject_mean.empty:
        for (sid, cond), sgroup in subject_mean.groupby(['id', 'condition']):
            if sgroup.empty:
                continue
            col = condition_colors.get(str(cond), None)
            plt.plot(sgroup['time'], sgroup['val'], alpha=0.2, linewidth=0.6, label=None, color=col)

    plt.xlabel('Time (s)')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _plot_contrast(contrast_group, contrast_subject, y_label, title, output_path):
    plt.figure(figsize=(8, 5))
    plt.fill_between(contrast_group['time'], contrast_group['mean'] - contrast_group['sem'], contrast_group['mean'] + contrast_group['sem'], linewidth=0, color='#333333', alpha=0.3, label='contrast (sem)')
    plt.plot(contrast_group['time'], contrast_group['mean'], color='#333333', label='contrast (mean)')

    for sid, sgroup in contrast_subject.groupby('id'):
        if sgroup.empty:
            continue
        plt.plot(sgroup['time'], sgroup['contrast'], alpha=0.2, linewidth=0.6, label=None, color='#333333')

    plt.xlabel('Time (s)')
    plt.ylabel(y_label)
    plt.title(title)
    plt.axvline(0, color='k', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _compute_subject_derivative(subject_mean):
    out = subject_mean.sort_values(['id', 'condition', 'time']).copy()
    out['dval_dt'] = np.nan

    for _, idx in out.groupby(['id', 'condition']).groups.items():
        group_idx = pd.Index(idx)
        t = out.loc[group_idx, 'time'].to_numpy(dtype=float)
        v = out.loc[group_idx, 'val'].to_numpy(dtype=float)

        if len(t) < 2:
            continue

        try:
            deriv = np.gradient(v, t)
        except Exception:
            dv = np.diff(v)
            dt = np.diff(t)
            deriv = np.concatenate([[np.nan], np.divide(dv, dt, out=np.full_like(dv, np.nan, dtype=float), where=dt != 0)])

        out.loc[group_idx, 'dval_dt'] = deriv

    return out


def _plot_derivative(cond_deriv, subj_deriv, condition_colors, title, output_path):
    plt.figure(figsize=(8, 5))
    for cond, cond_df in cond_deriv.groupby('condition'):
        col = condition_colors.get(str(cond), None)
        if col is not None:
            plt.fill_between(cond_df['time'], cond_df['mean'] - cond_df['sem'], cond_df['mean'] + cond_df['sem'], linewidth=0, color=col, alpha=0.3, label=f"{cond} (sem)")
            plt.plot(cond_df['time'], cond_df['mean'], color=col, label=f"{cond} (mean)")
        else:
            plt.fill_between(cond_df['time'], cond_df['mean'] - cond_df['sem'], cond_df['mean'] + cond_df['sem'], linewidth=0, alpha=0.3, label=f"{cond} (sem)")
            plt.plot(cond_df['time'], cond_df['mean'], label=f"{cond} (mean)")

    for (sid, cond), sgroup in subj_deriv.groupby(['id', 'condition']):
        if sgroup.empty:
            continue
        col = condition_colors.get(str(cond), None)
        plt.plot(sgroup['time'], sgroup['dval_dt'], alpha=0.2, linewidth=0.6, label=None, color=col)

    plt.xlabel('Time (s)')
    plt.ylabel('d(value)/dt')
    plt.title(title)
    plt.axvline(0, color='k', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# ---------------------------
# Main Pipeline
# ---------------------------

def main():
    today = datetime.datetime.today().strftime('%Y%m%d')
    
    parser = argparse.ArgumentParser(description='Preprocess timeseries data for paired-taVNS project')
    parser.add_argument('--data-dir', default=r"/Users/elise/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Desktop/paired-tavns-analysis/Data", help='Top-level data directory')
    parser.add_argument('--output-dir', default=r"/Users/elise/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Desktop/paired-tavns-analysis/analyzed-data", help='Output directory for processed data')
    parser.add_argument('--start-date', type=int, default=20250701, help='Start session (YYYYMMDD)')
    parser.add_argument('--end-date', type=int, default=np.inf, help='End session (YYYYMMDD)')
    parser.add_argument('--dry-run', action='store_true', help='List blocks that would be processed without writing output')
    parser.add_argument('--subject', help='Optional: only process this subject folder')
    parser.add_argument('--baseline-correct-method', choices=['zscore', 'delta', 'percent_change', 'none'], default='percent_change', help='Baseline correction method to apply to epochs')
    parser.add_argument('--skip-individual-plots', action='store_true', help='Skip per-subject individual trial plots to speed up runtime')
    args = parser.parse_args()

    data_dir = args.data_dir
    date_dir = os.path.join(args.output_dir, today)
    output_dir = os.path.join(date_dir, args.baseline_correct_method)
    start_date = args.start_date
    end_date = args.end_date
    dry_run = args.dry_run
    skip_individual_plots = args.skip_individual_plots
    os.makedirs(output_dir, exist_ok=True)

    # master accumulator for all block timeseries to compute experiment-level averages
    master_timeseries = []
    # condition colors (hex) — used for plotting
    condition_colors = {
        'sham': '#6dc8bf',
        'taVNS': '#f15a22'
    }
    features_by_experiment = {}
    all_metrics_rows = []
    for subject in os.listdir(data_dir):
        subject_path = os.path.join(data_dir, subject)
        if not os.path.isdir(subject_path) or subject.startswith("test") or subject.startswith('analyzed'):
            continue
        if args.subject and subject != args.subject:
            continue

        print(f"Processing {subject}...")
        
        for session in os.listdir(subject_path):
            session_path = os.path.join(subject_path, session)
            if not os.path.isdir(session_path) or not (start_date <= int(session) <= end_date):
                continue
            if (int(session) > 20250919 and int(session) < 20251118) or (int(session) == 20250829) or (int(session) == 20251121):
                # skip buggy sessions
                continue
            
            for block in os.listdir(session_path):
                block_path = os.path.join(session_path, block)
                if not os.path.isdir(block_path):
                    continue
                print(f"  Processing block {block}...")
                if dry_run:
                    print(f"    DRY RUN: would process {block_path}")
                    continue
                feature_data = []
                # accumulate timeseries for this block (list of long-form dfs)
                block_timeseries = []
                # check for stroop table in block directory
                stroop_trials = None
                stroop_path = os.path.join(block_path, f"{block}_stroopTrials.csv")
                if os.path.exists(stroop_path):
                    try:
                        stroop_trials = pd.read_csv(stroop_path)
                        # rename trial column to 'trial' if it exists
                        if 'trial_number' in stroop_trials.columns:
                            stroop_trials.rename(columns={'trial_number': 'trial'}, inplace=True)
                    except pd.errors.EmptyDataError:
                        print(f"Skipping stroop merge for {block}: empty file {stroop_path}")
                        stroop_trials = None
                    except pd.errors.ParserError as e:
                        print(f"Skipping stroop merge for {block}: parse error in {stroop_path} ({e})")
                        stroop_trials = None
                
                try:
                    cfg_path = os.path.join(block_path, f"{block}_config.json")
                    ts_path = os.path.join(block_path, f"{block}_tsData.csv")
                    if not os.path.exists(cfg_path):
                        print(f"Skipping {block_path}: missing config file {cfg_path}")
                        continue
                    if not os.path.exists(ts_path):
                        print(f"Skipping {block_path}: missing timeseries file {ts_path}")
                        continue

                    with open(cfg_path, 'r') as f:
                        block_cfg = json.load(f)
                    ts_data = pd.read_csv(ts_path)
                    if ts_data.empty:
                        print(f"No time series data found in {block_path}")
                        continue
                    
                    experiment_type = block_cfg.get('experiment', '')
                    if 'PLRT' in experiment_type:
                        pre, post = 0.2, 3
                    elif 'SCWT' in experiment_type:
                        pre, post = 1, 2
                    elif 'StroopSquared' in experiment_type:
                        pre, post = 1, 2
                    else:
                        continue

                    df_events = prepare_block_events(ts_data, experiment_type)
                    if df_events.empty:
                        continue

                    # Only use events present in df_events for epoching
                    marker = df_events['event'].unique().tolist()

                    # Collect both timeseries (long-form) and feature-level data
                    for event in marker:
                        event_df = df_events[df_events['event'] == event]
                        trial_data = epoch_by_event(ts_data, event, event_df, pre_event_dur=pre, post_event_dur=post, block_cfg=block_cfg, baseline_method=args.baseline_correct_method)
                        if trial_data is not None:
                            ts, feat = trial_data
                            # accumulate timeseries for block-level averaging
                            if ts is not None and not ts.empty:
                                block_timeseries.append(ts)
                            # feature rows
                            if feat is not None and not feat.empty:
                                feature_data.append(feat)

                    # If we collected any feature rows, concatenate and save
                    if feature_data:
                        feature_df = pd.concat(feature_data, ignore_index=True)
                        if stroop_trials is not None:
                            # Merge with stroop trials if available
                            feature_df = feature_df.merge(stroop_trials, on=['trial'], how='left')
                        exp_name = str(block_cfg.get('experiment', 'unknown')).strip() if block_cfg else 'unknown'
                        features_by_experiment.setdefault(exp_name, []).append(feature_df)

                        # # Compute condition averages per (id, block, condition)
                        # # choose numeric feature columns to average
                        # numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
                        # # exclude 'trial' from aggregated means (not a signal feature)
                        # if 'trial' in numeric_cols:
                        #     numeric_cols.remove('trial')

                        # if len(numeric_cols) > 0:
                        #     agg_block = feature_df.groupby(['id', 'block', 'condition'])[numeric_cols].mean().reset_index()
                        #     agg_subject = feature_df.groupby(['id', 'condition'])[numeric_cols].mean().reset_index()
                        #     agg_group = feature_df.groupby(['condition'])[numeric_cols].mean().reset_index()
                            
                            


                    # -------------------------
                    # Block-level timeseries averaging and plotting
                    # -------------------------
                    if block_timeseries:
                        block_ts_df = pd.concat(block_timeseries, ignore_index=True)
                        # ensure numeric types
                        block_ts_df['time'] = pd.to_numeric(block_ts_df['time'], errors='coerce')
                        block_ts_df['value'] = pd.to_numeric(block_ts_df['value'], errors='coerce')

                        # # save a CSV per block with the raw concatenated timeseries
                        # block_out_dir = os.path.join(output_dir, 'timeseries_by_block')
                        # os.makedirs(block_out_dir, exist_ok=True)
                        # block_csv = os.path.join(block_out_dir, f"{block}_timeseries_raw.csv")
                        # block_ts_df.to_csv(block_csv, index=False)

                        # append block-level timeseries to master list for experiment-level aggregation
                        master_timeseries.append(block_ts_df)

                except Exception as e:
                    tb = sys.exc_info()[2]
                    stack = traceback.extract_tb(tb)
                    func_name = stack[-1].name if stack else '<unknown>'
                    line_no = stack[-1].lineno if stack else '<unknown>'
                    print(f"Error processing {block_path}: {e} (line {line_no} in {func_name})")
                    print(traceback.format_exc())
        
    # ----------------------------
    # After main run: compute experiment-level overlays (mirror plotTimeSeries.R)
    # ----------------------------
    try:
        if master_timeseries:
            all_ts = pd.concat(master_timeseries, ignore_index=True)
            # coerce types
            all_ts['time'] = pd.to_numeric(all_ts['time'], errors='coerce')
            all_ts['value'] = pd.to_numeric(all_ts['value'], errors='coerce')

            # remove practice condition if present
            all_ts = all_ts[all_ts['condition'] != 'practice'] if 'condition' in all_ts.columns else all_ts

            # loop over experiments, events, signals similar to R script
            for exp in all_ts['experiment'].dropna().unique():
                exp_df = all_ts[all_ts['experiment'] == exp]
                exp_out_dir = os.path.join(output_dir, f"timeseries_plots_{exp}")
                os.makedirs(exp_out_dir, exist_ok=True)

                event_values = np.asarray(exp_df['event'].dropna().astype(str), dtype=object)
                for ev in np.unique(event_values):
                    ev_out_dir = os.path.join(exp_out_dir, f"{ev}")
                    os.makedirs(ev_out_dir, exist_ok=True)
                    ev_df = exp_df[exp_df['event'] == ev]
                    signal_values = np.asarray(ev_df['signal_type'].dropna().astype(str), dtype=object)
                    for sig in np.unique(signal_values):
                        sig_df = ev_df[ev_df['signal_type'] == sig]
                        

                        # per-subject mean trace (id x condition x time)
                        subject_df = sig_df.groupby(['experiment', 'signal_type', 'event', 'id', 'condition', 'time'])['value']
                        subject_mean = subject_df.mean().reset_index().rename(columns={'value': 'val'})

                        # Collect metrics from average timeseries per subject/condition
                        metrics_rows = []
                        if not subject_mean.empty:
                            for (sid, cond), sgroup in subject_mean.groupby(['id', 'condition']):
                                if sgroup.empty:
                                    continue
                                m = compute_timeseries_metrics(sgroup[['time', 'val']])
                                for k, v in m.items():
                                    metrics_rows.append({
                                        'expt': exp,
                                        'event': ev,
                                        'signal': sig,
                                        'id': sid,
                                        'condition': cond,
                                        'metric': k,
                                        'value': v,
                                    })
                        if metrics_rows:
                            all_metrics_rows.extend(metrics_rows)

                        # condition mean and sem across subjects
                        condition_mean = subject_mean.groupby(['experiment', 'signal_type', 'event', 'condition', 'time']).agg(
                            n_subjects=('id', 'nunique'),
                            mean=('val', 'mean'),
                            std=('val', 'std'),
                        ).reset_index()
                        condition_mean = _add_sem(condition_mean, 'val')

                        
                    
                        rt_means = None
                        if str(exp) == 'SCWT' and str(sig) == 'pupilDiameter' and 'rt' in sig_df.columns:
                            try:
                                rt_trials = sig_df[['condition', 'id', 'datetime', 'trial', 'rt']].dropna(subset=['rt'])
                                if not rt_trials.empty:
                                    rt_trials = rt_trials.drop_duplicates(subset=['condition', 'id', 'datetime', 'trial'])
                                    rt_means = rt_trials.groupby('condition')['rt'].mean()
                            except Exception:
                                rt_means = None

                        plotfile = os.path.join(ev_out_dir, f"{exp}_{ev}_{sig}.svg")
                        _plot_condition_mean(
                            subject_mean=subject_mean,
                            condition_mean=condition_mean,
                            condition_colors=condition_colors,
                            y_label=args.baseline_correct_method,
                            title=f"{exp} : {ev} : {sig}",
                            output_path=plotfile,
                            rt_means=rt_means,
                        )

                        # ---------------------------------------
                        # Contrast: taVNS - sham (by subject, then mean+SEM)
                        # ---------------------------------------
                        try:
                            # Build wide condition table per subject/time
                            wide = subject_mean.pivot_table(index=['id', 'time'], columns='condition', values='val')
                            # Ensure both conditions are present
                            if {'taVNS', 'sham'}.issubset(set(wide.columns)):
                                wide = wide.reset_index()
                                wide['contrast'] = wide['taVNS'] - wide['sham']
                                contrast_subject = wide[['id', 'time', 'contrast']]

                                # Group mean and SEM across subjects
                                contrast_group = contrast_subject.groupby('time').agg(
                                    n_subjects=('id', 'nunique'),
                                    mean=('contrast', 'mean'),
                                    std=('contrast', 'std'),
                                ).reset_index()
                                contrast_group = _add_sem(contrast_group, 'contrast')

                                cplotfile = os.path.join(ev_out_dir, f"{exp}_{ev}_{sig}_contrast.svg")
                                _plot_contrast(
                                    contrast_group=contrast_group,
                                    contrast_subject=contrast_subject,
                                    y_label=args.baseline_correct_method,
                                    title=f"{exp} : {ev} : {sig} : taVNS - sham",
                                    output_path=cplotfile,
                                )
                        except Exception:
                            # Keep overlays robust; skip contrast if any issue
                            pass

                        # ---------------------------------------
                        # First derivative plots (pupilDiameter only)
                        # ---------------------------------------
                        if str(sig) == 'pupilDiameter' and not subject_mean.empty:
                            subj_deriv = _compute_subject_derivative(subject_mean)

                            # Condition mean and SEM over subjects
                            cond_deriv = subj_deriv.groupby(['condition', 'time']).agg(
                                n_subjects=('id', 'nunique'),
                                mean=('dval_dt', 'mean'),
                                std=('dval_dt', 'std'),
                            ).reset_index()
                            cond_deriv = _add_sem(cond_deriv, 'dval_dt')

                            dplotfile = os.path.join(ev_out_dir, f"{exp}_{ev}_{sig}_d1.svg")
                            _plot_derivative(
                                cond_deriv=cond_deriv,
                                subj_deriv=subj_deriv,
                                condition_colors=condition_colors,
                                title=f"{exp} : {ev} : {sig} (first derivative)",
                                output_path=dplotfile,
                            )
                        # ---------------------------------------
                        # Diagnostics: per-subject trial counts
                        # ---------------------------------------
                        try:
                            pairs = sig_df[['id', 'condition', 'datetime', 'event','trial']].dropna()
                            pairs['pair'] = pairs['datetime'].astype(str) + '|' + pairs['trial'].astype(str)
                            trial_counts = pairs.groupby(['event','id', 'condition'])['pair'].nunique().reset_index().rename(columns={'pair': 'n_trials'})

                            diag_dir = os.path.join(output_dir, 'diagnostics')
                            os.makedirs(diag_dir, exist_ok=True)
                            diag_file = os.path.join(diag_dir, f"trial_counts_{exp}_{ev}_{sig}.csv")
                            trial_counts.to_csv(diag_file, index=False)

                            
                        except Exception:
                            # Keep pipeline robust; diagnostics are optional
                            pass
                        # ----------------------------
                        # Individual subject trial plots (saved per event)
                        # ----------------------------
                        if not skip_individual_plots:
                            indiv_dir = os.path.join(ev_out_dir, 'individual_trials', f"{sig}")
                            os.makedirs(indiv_dir, exist_ok=True)

                            subject_ids = pd.unique(sig_df['id'].dropna())
                            for sid in subject_ids:
                                subj_df = sig_df[sig_df['id'] == sid]
                                subj_mean = subject_mean[subject_mean['id'] == sid]
                                # Count unique trials (by datetime x trial) for this subject
                                try:
                                    trial_pairs = subj_df[['datetime', 'trial']].dropna().drop_duplicates()
                                    n_trials = int(trial_pairs.shape[0])
                                except Exception:
                                    n_trials = None
                                plt.figure(figsize=(8,5))
                                subject_conds = pd.unique(subj_df['condition'].dropna())
                                for cond in subject_conds:
                                    cond_df = subj_df[subj_df['condition'] == cond]
                                    col = condition_colors.get(str(cond), None)
                                    # plot subject mean
                                    cond_mean = subj_mean[subj_mean['condition'] == cond]
                                    if not cond_mean.empty:
                                        plt.plot(cond_mean['time'], cond_mean['val'], color=col, label=f"{cond} (mean)")
                                    for block in cond_df['datetime'].dropna().unique():
                                        block_df = cond_df[cond_df['datetime'] == block]
                                        for trial in block_df['trial'].dropna().unique():
                                            trial_df = block_df[block_df['trial'] == trial]
                                            plt.plot(trial_df['time'], trial_df['value'], alpha=0.2, linewidth=0.6, color=col)

                                plt.xlabel('Time (s)')
                                plt.ylabel(args.baseline_correct_method)
                                if n_trials is not None:
                                    plt.title(f"{exp} : {ev} : {sig} : Subject {sid} (n trials={n_trials})")
                                else:
                                    plt.title(f"{exp} : {ev} : {sig} : Subject {sid}")
                                plt.axvline(0, color='k', linestyle='--')
                                plt.legend()
                                plt.tight_layout()
                                indiv_plotfile = os.path.join(indiv_dir, f"{exp}_{ev}_{sig}_{sid}.svg")
                                plt.savefig(indiv_plotfile, dpi=300)
                                plt.close()
                # ---------------------------------------
                # SCWT congruency contrast: (incongruent - congruent) per condition
                # ---------------------------------------
                # if str(exp).upper() == 'SCWT':
                #     cong_dir = os.path.join(exp_out_dir, 'congruency_contrast')
                #     os.makedirs(cong_dir, exist_ok=True)
                #     for sig in exp_df['signal_type'].dropna().unique():
                #         sig_all = exp_df[exp_df['signal_type'] == sig]
                #         # Identify congruent/incongruent correct cue events
                #         cong_mask = sig_all['event'].astype(str).str.contains('cue.*congruent.*correct', case=False, na=False)
                #         incong_mask = sig_all['event'].astype(str).str.contains('cue.*incongruent.*correct', case=False, na=False)
                #         cong_df = sig_all[cong_mask]
                #         incong_df = sig_all[incong_mask]
                #         if cong_df.empty or incong_df.empty:
                #             continue

                #         # Subject-level mean traces for congruent and incongruent
                #         subj_cong = cong_df.groupby(['id', 'condition', 'time'])['value'].mean().reset_index().rename(columns={'value': 'congruent'})
                #         subj_incong = incong_df.groupby(['id', 'condition', 'time'])['value'].mean().reset_index().rename(columns={'value': 'incongruent'})

                #         # Inner join by id/condition/time to ensure aligned samples
                #         subj_merge = pd.merge(subj_incong, subj_cong, on=['id', 'condition', 'time'], how='inner')
                #         if subj_merge.empty:
                #             continue
                #         subj_merge['val'] = subj_merge['incongruent'] - subj_merge['congruent']

                #         # Compute metrics from contrast timeseries per subject/condition
                #         metrics_rows = []
                #         for (sid, cond), sgroup in subj_merge.groupby(['id', 'condition']):
                #             m = compute_timeseries_metrics(sgroup[['time', 'val']])
                #             for k, v in m.items():
                #                 metrics_rows.append({
                #                     'expt': exp,
                #                     'event': 'congruency_contrast',
                #                     'signal': sig,
                #                     'id': sid,
                #                     'condition': cond,
                #                     'metric': k,
                #                     'value': v,
                #                 })

                #         if metrics_rows:
                #             metrics_df = pd.DataFrame(metrics_rows)
                #             metrics_dir = os.path.join(output_dir, 'metrics_by_timeseries')
                #             os.makedirs(metrics_dir, exist_ok=True)
                #             metrics_csv = os.path.join(metrics_dir, 'timeseries_metrics.csv')
                #             metrics_df.to_csv(metrics_csv, index=False, mode='a', header=not os.path.exists(metrics_csv))

                #         # Group condition mean and SEM across subjects for contrast
                #         cond_mean = subj_merge.groupby(['condition', 'time']).agg(
                #             n_subjects=('id', 'nunique'),
                #             mean=('val', 'mean'),
                #             sem=('val', lambda x: x.std(ddof=0) / np.sqrt(x.nunique() if x.nunique() > 0 else 1))
                #         ).reset_index()

                #         # Plot contrast per condition with subject overlays
                #         plt.figure(figsize=(6, 4.5))
                #         plt.axvline(0, color='k', linestyle='--')
                #         for cond, cdf in cond_mean.groupby('condition'):
                #             col = condition_colors.get(str(cond), None)
                #             if col is not None:
                #                 plt.fill_between(cdf['time'], cdf['mean'] - cdf['sem'], cdf['mean'] + cdf['sem'], linewidth=0, color=col, alpha=0.3, label=f"{cond} (sem)")
                #                 plt.plot(cdf['time'], cdf['mean'], color=col, label=f"{cond} (mean)")
                #             else:
                #                 plt.fill_between(cdf['time'], cdf['mean'] - cdf['sem'], cdf['mean'] + cdf['sem'], linewidth=0, alpha=0.3, label=f"{cond} (sem)")
                #                 plt.plot(cdf['time'], cdf['mean'], label=f"{cond} (mean)")

                #         # overlay subject contrast traces
                #         for (sid, cond), sgroup in subj_merge.groupby(['id', 'condition']):
                #             col = condition_colors.get(str(cond), None)
                #             plt.plot(sgroup['time'], sgroup['val'], alpha=0.25, linewidth=0.6, label=None, color=col)

                #         plt.xlabel('Time (s)')
                #         plt.ylabel(args.baseline_correct_method)
                #         plt.title(f"{exp} : congruency_contrast : {sig}")
                #         plt.legend()
                #         plt.tight_layout()
                #         out_file = os.path.join(cong_dir, f"{exp}_congruency_contrast_{sig}.svg")
                #         plt.savefig(out_file, dpi=300)
                #         plt.close()

        if features_by_experiment:
            exp_feat_dir = os.path.join(output_dir, 'features_by_experiment')
            os.makedirs(exp_feat_dir, exist_ok=True)
            for exp_name, frames in features_by_experiment.items():
                if not frames:
                    continue
                csv_file = os.path.join(exp_feat_dir, f"features-table_{exp_name}.csv")
                pd.concat(frames, ignore_index=True).to_csv(csv_file, index=False)

        if all_metrics_rows:
            metrics_dir = os.path.join(output_dir, 'metrics_by_timeseries')
            os.makedirs(metrics_dir, exist_ok=True)
            metrics_csv = os.path.join(metrics_dir, 'timeseries_metrics.csv')
            pd.DataFrame(all_metrics_rows).to_csv(metrics_csv, index=False)
    except Exception:
        # don't crash after main; print traceback for debugging
        print("Error computing experiment-level overlays:")
        traceback.print_exc()

    print(f"Data successfully exported to {output_dir}")


if __name__ == "__main__":
    main()