import pandas as pd
import numpy as np

# --- 1. Load Your Data ---

# b) The Unix timestamp for this event from your Pupil Player annotation.
SYNC_EVENT_UNIX_TIME = 1761685590.16953  # Replace with your actual value

# c) The monotonic timestamp for this same event from your host-logged file.
SYNC_EVENT_MONOTONIC_TIME = 11018.984


# --- 3. Calculate the LSL Time Correction Offset ---

# The offset is the consistent difference between the corrected LSL time and the raw LSL time.
# We can calculate it from your pupil data.
# Note: This assumes the raw LSL timestamp was also saved, as in the StreamRecorder.
# If not, you may need to find the first pupil sample's unix time and its corrected time.
# For this example, let's assume the offset was a known value. A better way is to find it from the data.

# Let's find the first pupil sample that has a unix timestamp close to our event
# This is a bit tricky without the original unix times in the pupil file.
# A simpler way: The LSL offset is the mean difference between the corrected time and the raw LSL time.
# Let's assume you can calculate or find this value.
# For this example, let's use a placeholder.
# In a real scenario, you'd derive this from your data.
LSL_OFFSET_PUPIL_TO_HOST = -2015.349955


# --- 4. Bridge the Clocks ---

# a) Calculate what the LSL-Corrected time of your sync event SHOULD have been.
# This brings the Unix annotation onto the same timeline as your pupil data.
inferred_lsl_corrected_time = SYNC_EVENT_UNIX_TIME + LSL_OFFSET_PUPIL_TO_HOST

# b) Now, calculate the offset between the LSL-Corrected clock and the Host Monotonic clock.
final_offset = inferred_lsl_corrected_time - SYNC_EVENT_MONOTONIC_TIME

print(f"Inferred LSL-Corrected time for sync event: {inferred_lsl_corrected_time}")
print(f"Recorded Host Monotonic time for sync event: {SYNC_EVENT_MONOTONIC_TIME}")
print(f"--> Final offset to align clocks: {final_offset}")


# --- 5. Apply the Final Offset to All Your Events ---

# This brings all your host-logged events onto the same timeline as the pupil data.
host_events_df['corrected_timestamp'] = host_events_df['Timestamp'] + final_offset

# Now, 'corrected_timestamp' in host_events_df and 'corrected_lsl_timestamp'
# in pupil_df are aligned and can be used together for analysis.

print("\nOriginal vs. Corrected Event Timestamps:")
print(host_events_df[['Event', 'Timestamp', 'corrected_timestamp']].head())
