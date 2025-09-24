import nidaqmx
from nidaqmx.constants import LineGrouping
import time

reps = 5 # number of repetitions
bits = 3
codes = 2 ** bits # number of codes
trig_dur = 1 # trigger duration in seconds
iti = 1 # inter-trial interval in seconds


# 1->2 on MSI
# 2->4 on MSI
# 3->6 on MSI
# 4->16 on MSI
# 5->32 on MSI
# 6->64 on MSI
# 7->128 on MSI

def int_to_bool_list(value):
    """Convert an integer to a list of booleans (bit0 -> line1, bit1.)."""
    return [(value >> bit) & 1 == 1 for bit in range(bits)]

with nidaqmx.Task() as do_task:
    # Create channel with 3 lines (1:3) for output
    do_task.do_channels.add_do_chan("Dev1/port1/line1:3", line_grouping=LineGrouping.CHAN_PER_LINE)

    try:
        for i in range(reps):
            for j in range(codes):
                trig_value = j  # generates trigger code for bit j: 0, 1, 2, 3, 4, 5, 6, 7
                print('Trig =', trig_value)
                bool_array = int_to_bool_list(trig_value)
                print('Bool array =', bool_array)
                do_task.write(bool_array)
                time.sleep(trig_dur)
                # Reset: set all lines low
                do_task.write([False] * bits)
                time.sleep(iti)
    except KeyboardInterrupt:
        print('Trigger output aborted by user.')

