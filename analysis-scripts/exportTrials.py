import os
import pandas as pd
import json
import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt
from based_noise_blinks_detection import detect_blinks
import neurokit2 as nk
import datetime
import seaborn as sns

def epoch_by_event(block_data, event, event_df, pre_event_dur=5, post_event_dur=5, block_cfg=None):
    """Returns a tidy dataframe with trial-epoch data for the specified event."""
    event_inds =event_df.index            
    event_times = block_data.loc[event_inds, 'Timestamps']
    if event_times.empty:
        return None

    event_str = [event]
    if len(event_str) != 1:
        raise ValueError("Multiple unique event strings found, expected one.")
    
    dt = block_data['Timestamps'].diff().mean()
    pre_samples = int(pre_event_dur / dt)
    post_samples = int(post_event_dur / dt)
    
    # find skin conductance events
    eda_thresh = 0.1 * np.max(block_data['SCR'])

    timeseries_data = []
    feature_data = []
    for i, (event_idx, event_row) in enumerate(event_df.iterrows()):
        start_ind = event_idx - pre_samples
        end_ind = event_idx + post_samples
        if start_ind < 0 or end_ind >= len(block_data):
            continue
        
        rel_time = block_data['Timestamps'].iloc[start_ind:end_ind].to_numpy() - block_data['Timestamps'].iloc[event_idx]
        for sig in block_data.columns.difference(['Timestamps', 'Event', 'nSeq']):
            values = block_data[sig].iloc[start_ind:end_ind].to_numpy()
            
            # First, z-score the trial using pre-event-dur.
            BL_mask = (rel_time >= -0.5) & (rel_time < 0)
            BL_mean = values[BL_mask].mean()
            BL_std = values[BL_mask].std()
            if BL_std == 0:
                BL_std = 1
            
            values = (values - BL_mean) / BL_std
            features = extract_features(values, rel_time, sig)

            
            timeseries_data.append(pd.DataFrame({
                'id': block_cfg.get('participant_ID'),
                'order': block_cfg.get('order'),
                'datetime': block_cfg.get('datetime'),
                'condition': block_cfg.get('condition'),
                'experiment': block_cfg.get('experiment'),
                'block': block_cfg.get('block_no'),
                'trial': event_df['trial'].iloc[i],
                'time': rel_time.round(3),
                'signal_type': sig,
                'event': event_str[0].strip(''),
                'value': values.round(3)
            }))
            
            feature_data.append(pd.DataFrame([{
                'id': block_cfg.get('participant_ID'),
                'order': block_cfg.get('order'),
                'datetime': block_cfg.get('datetime'),
                'condition': block_cfg.get('condition'),
                'experiment': block_cfg.get('experiment'),
                'block': block_cfg.get('block_no'),
                'trial': event_df['trial'].iloc[i],
                'event': event_str[0].strip(''),
                'signal_type': sig,
                **features
            }]))

    return pd.concat(timeseries_data, ignore_index=True), pd.concat(feature_data, ignore_index=True)

def extract_features(trial_data, time, signal, thresh=0):
    """Extracts features from the trial timeseries."""
    if trial_data is None:
        return None
    

    post_event_mask = (time >= 0)
    trial_data = trial_data[post_event_mask]
    time = time[post_event_mask]
    
    # if signal == 'pupilDiameter' or signal == 'EDA': "Detect event-related pupil response"
    max_value = trial_data.max()
    min_value = trial_data.min()
    peak_time = time[trial_data.argmax()]
    trough_time = time[trial_data.argmin()]
    dt = int(np.mean(np.diff(time)))
    auc = np.trapezoid(trial_data, time, dx=dt)
    abs_auc = np.trapezoid(np.abs(trial_data), time, dx=dt)

    features = {
        'mean': trial_data.mean(),
        'std': trial_data.std(),
        'min': min_value,
        'max': max_value,
        'peak_time': peak_time,
        'trough_time': trough_time,
        'auc': auc,
        'abs_auc': abs_auc,
    }
        
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

# ---------------------------
# Main Pipeline
# ---------------------------

def main():
    today = datetime.datetime.today().strftime('%Y%m%d')
    data_dir = "/Users/elise/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Desktop/pairedStroop/Data"
    output_dir = "/Users/elise/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Desktop/pairedStroop/analyzed-data"
    output_dir = os.path.join(output_dir, today)
    
    os.makedirs(output_dir, exist_ok=True)

    start_date = 20250501
    end_date = 20250701
    for subject in os.listdir(data_dir):
        subject_path = os.path.join(data_dir, subject)
        if not os.path.isdir(subject_path) or subject.startswith("test"):
            continue
        print(f"Processing {subject}...")
        
        for session in os.listdir(subject_path):
            session_path = os.path.join(subject_path, session)
            if not os.path.isdir(session_path) or not (start_date <= int(session) <= end_date):
                continue
            

            for block in os.listdir(session_path):
                block_path = os.path.join(session_path, block)
                if not os.path.isdir(block_path):
                    continue
                
                feature_data = []
                # check for stroop table in block directory
                if os.path.exists(os.path.join(block_path, f"{block}_stroopTrials.csv")):
                    # load stroop trials
                    stroop_trials = pd.read_csv(os.path.join(block_path, f"{block}_stroopTrials.csv"))
                    # rename trial column to 'trial' if it exists
                    if 'trial_number' in stroop_trials.columns:
                        stroop_trials.rename(columns={'trial_number': 'trial'}, inplace=True)
                
                try:
                    block_cfg = json.load(open(os.path.join(block_path, f"{block}_config.json"), 'r'))
                    ts_data = pd.read_csv(os.path.join(block_path, f"{block}_tsData.csv"))
                    if ts_data.empty:
                        print(f"No time series data found in {block_path}")
                        continue
                    
                    experiment_type = block_cfg.get('experiment', '')
                    if 'PLRT' in experiment_type or 'SCWT' in experiment_type:
                        pre, post = 2, 5
                    elif 'StroopSquared' in experiment_type:
                        pre, post = 1, 2
                    else:
                        continue

                    df_events = pd.DataFrame({'event': ts_data['Event'].dropna()})
                    # Only filter events for SCWT or StroopSquared experiments
                    if experiment_type in ['SCWT', 'StroopSquared']:
                        # remove cue/stimulus trials that are not proceeded by correct response
                        # Filter events to only those containing "stim", "cue", or "response"
                        df_events = df_events[df_events['event'].str.contains('stim|cue|response', case=False, na=False)]

                        for pos, (idx, event) in enumerate(df_events['event'].items()):
                            if 'stim' in event.lower() or 'cue' in event.lower():
                                if pos + 1 < len(df_events):
                                    next_event = df_events['event'].iloc[pos + 1]
                                    response = next_event.split('_')[-1]  # Get the last part after underscore
                                    df_events.at[idx, 'event'] = f"{event}_{response}"
                                else: #delete the event
                                    df_events.drop(idx, inplace=True)
                            
                        response_mask = df_events['event'].str.contains('response_', case=False, na=False)
                        df_events = df_events.loc[~response_mask]
                        df_events['trial'] = np.arange(1, len(df_events) + 1, dtype=int)
                        
                        # df_events.loc[stim_cue_mask, 'trial'] = np.arange(1, stim_cue_mask.sum() + 1, dtype=int)
                        # df_events['trial'] = df_events['trial'].ffill()
                        # # Check if the next event is a "correct" response
                        # correct_next_mask = stim_cue_mask & df_events['event'].shift(-1).str.contains('correct', case=False, na=False)
                        # df_events = df_events[correct_next_mask | df_events['event'].str.contains('response', case=False, na=False)]
                        # add trial number to df_events

                    
                    # Only use events present in df_events for epoching
                    marker = df_events['event'].unique().tolist()

                    timeseries_data = []
                    for event in marker:
                        # Find indices in ts_data that match the event and are in df_events
                        event_df = df_events[df_events['event'] == event]
                        trial_data = epoch_by_event(ts_data, event, event_df, pre_event_dur=pre, post_event_dur=post, block_cfg=block_cfg)
                        ts, feat = trial_data
                        if ts is not None and not ts.empty:
                            timeseries_data.append(ts)
                        if feat is not None and not feat.empty:
                            feature_data.append(feat)

                    if timeseries_data:
                        timeseries_df = pd.concat(timeseries_data, ignore_index=True)
                       
                        csv_file = os.path.join(output_dir, "epochs-table.csv")
                        timeseries_df.to_csv(csv_file, index=False, mode="a", header=not os.path.exists(csv_file))

                    if feature_data:
                        # check for 
                        feature_df = pd.concat(feature_data, ignore_index=True)
                        if stroop_trials is not None:
                            # Merge with stroop trials if available
                            feature_df = feature_df.merge(stroop_trials, on=['trial'], how='left')
                        csv_file = os.path.join(output_dir, "features-table.csv")
                        feature_df.to_csv(csv_file, index=False, mode="a", header=not os.path.exists(csv_file))

                except Exception as e:
                    print(f"Error processing {block_path}: {e}")

    print(f"Data successfully exported to {output_dir}")


if __name__ == "__main__":
    main()