#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This adaptation to Python was made with the supervision and encouragement of Upamanyu Ghose
For more information about this adaptation and for more Python solutions, don't hesitate to contact him:
Email: titoghose@gmail.com
Github code repository: github.com/titoghose

Note: Please install the numpy python library to use this code:
		sudo pip install numpy (Python 2)
		sudo pip3 install numpy (Python 3)
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def diff(series):
	"""
	Python implementation of matlab's diff function
	"""
	return series[1:] - series[:-1]

def smooth(x, window_len):
	"""
	Python implementation of matlab's smooth function
	"""

	window_len = int(window_len)
 
	if window_len < 3:
		return x

	# Window length must be odd
	if window_len % 2 == 0:
		window_len += 1

	w = np.ones(window_len)
	y = np.convolve(w, x, mode='valid') / len(w)
	y = np.hstack((x[:window_len//2], y, x[len(x)-window_len//2:]))

	for i in range(0, window_len//2):
		y[i] = np.sum(y[0 : i+i]) / ((2*i) + 1)

	for i in range(len(x)-window_len//2, len(x)):
		y[i] = np.sum(y[i - (len(x) - i - 1) : i + (len(x) - i - 1)]) / ((2*(len(x) - i - 1)) + 1)

	return y

def detect_blinks(pupil_size, sampling_freq):
	"""
	Function to find blinks and return blink onset and offset indices
	Adapted from: R. Hershman, A. Henik, and N. Cohen, “A novel blink detection method based on pupillometry noise,” Behav. Res. Methods, vol. 50, no. 1, pp. 107–114, 2018.

	Input:
		pupil_size          : [numpy array/list] of pupil size data for left/right eye
		sampling_freq       : [float] sampling frequency of eye tracking hardware (default = 1000 Hz)
	Output:
		blinks              : [dictionary] {"blink_onset", "blink_offset"} containing numpy array/list of blink onset and offset indices
	"""
	sampling_interval = 1000 // sampling_freq
	concat_gap_interval = 100

	blink_onset = []
	blink_offset = []
	blinks = {"blink_onset": blink_onset, "blink_offset": blink_offset}
	pupil_size = np.asarray(pupil_size)
 
	#### ADDED THIS PART BECAUSE PUPIL LABS DOES NOT AUTOMATICALLY ZERO OUT BLINKS
	pupil_smooth = pd.Series(pupil_size).rolling(window=3, center=True, min_periods=1).mean()
	pupil_diff = np.diff(pupil_smooth, prepend=0)  # Prepend 0 to maintain the same length as pupil_size
	threshold = np.std(pupil_diff) * 2  # Adjust sensitivity
	pupil_size[np.abs(pupil_diff) > threshold] = 0
 
	# plot raw pupil data and velocity
	# plt.figure(figsize=(12, 6))
	# plt.plot(pupil_size)
	# plt.plot(pupil_diff)
	# plt.axhline(threshold)
	# plt.show()
 
	#### END OF ADDED CODE

	missing_data = np.array(pupil_size == 0, dtype="float32")
	difference = diff(missing_data)
	blink_onset = np.where(difference == 1)[0]
	blink_offset = np.where(difference == -1)[0] + 1

	length_blinks = len(blink_offset) + len(blink_onset)

	# Edge Case 1: there are no blinks
	if (length_blinks == 0):
		return blinks

	# Edge Case 2: the data starts with a blink.
	if ((len(blink_onset) < len(blink_offset)) or ((len(blink_onset) == len(blink_offset)) and (blink_onset[0] > blink_offset[0]))) and pupil_size[0] == 0:
		blink_onset = np.hstack((0, blink_onset))

	# Edge Case 3: the data ends with a blink.
	if (len(blink_offset) < len(blink_onset)) and pupil_size[-1] == 0:
		blink_offset = np.hstack((blink_offset, len(pupil_size) - 1))

	# Smoothing the data
	ms_4_smoothing = 100
	samples2smooth = ms_4_smoothing // sampling_interval
	smooth_pupil_size = np.array(smooth(pupil_size, samples2smooth), dtype='float32')
 
	smooth_pupil_size[np.where(smooth_pupil_size == 0)[0]] = float('nan')
	smooth_pupil_size_diff = diff(smooth_pupil_size)

	monotonically_dec = smooth_pupil_size_diff <= 0
	monotonically_inc = smooth_pupil_size_diff >= 0

	# Update blink onsets and offsets using smoothed data
	for i in range(len(blink_onset)):
		if blink_onset[i] != 0:
			j = blink_onset[i] - 1
			while j > 0 and monotonically_dec[j]:
				j -= 1
			blink_onset[i] = j + 1

		if blink_offset[i] != len(pupil_size) - 1:
			j = blink_offset[i]
			while j < len(monotonically_inc) and monotonically_inc[j]:
				j += 1
			blink_offset[i] = j

	# Removing duplications
	c = np.empty((len(blink_onset) + len(blink_offset),), dtype=blink_onset.dtype)
	c[0::2] = blink_onset
	c[1::2] = blink_offset
	c = list(c)

	i = 1
	while i < len(c) - 1:
		if c[i+1] - c[i] <= concat_gap_interval:
			c[i:i+2] = []
		else:
			i += 2

	temp = np.reshape(c, (-1, 2), order='C')

	blinks["blink_onset"] = (temp[:, 0]) # * sampling_interval) + sampling_interval
	blinks["blink_offset"] = (temp[:, 1]) # * sampling_interval) + sampling_interval
 
	# # Plot raw and smoothed pupil data with blink onsets and offsets
	# plt.figure(figsize=(12, 6))
	# plt.plot(pupil_size, label='Raw Pupil Data', color='blue', linewidth=3)
	# plt.plot(smooth_pupil_size, label='Smoothed Pupil Data', color='orange', linewidth=2)
	
	# # Plot each blink onset as a vertical green line
	# for idx, onset in enumerate(blinks["blink_onset"]):
	# 	if idx == 0:
	# 		plt.axvline(x=onset, color='green', linestyle='--', label='Blink Onset')
	# 	else:
	# 		plt.axvline(x=onset, color='green', linestyle='--')
	
	# # Plot each blink offset as a vertical red line
	# for idx, offset in enumerate(blinks["blink_offset"]):
	# 	if idx == 0:
	# 		plt.axvline(x=offset, color='red', linestyle='--', label='Blink Offset')
	# 	else:
	# 		plt.axvline(x=offset, color='red', linestyle='--')
			
	# plt.xlabel('Index')
	# plt.ylabel('Pupil Size')
	# plt.legend()
	# plt.show()
	
	return blinks

