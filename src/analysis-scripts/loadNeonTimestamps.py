import os
import numpy as np
import msgpack

rec_dir = r"C:\path\to\your\recording"  # change to your recording dir
ts_path = os.path.join(rec_dir, "gaze_timestamps.npy")
if os.path.exists(ts_path):
    timestamps = np.load(ts_path)
else:
    # fallback: read timestamps from gaze.pldata
    pldata_path = os.path.join(rec_dir, "gaze.pldata")
    timestamps = []
    with open(pldata_path, "rb") as fh:
        unpacker = msgpack.Unpacker(fh, use_list=False, strict_map_key=False)
        for topic, payload in unpacker:
            datum = msgpack.unpackb(payload, strict_map_key=False)
            timestamps.append(datum["timestamp"])
    timestamps = np.array(timestamps, dtype=float)
print("Loaded", timestamps.size, "timestamps. First:", timestamps[:5])