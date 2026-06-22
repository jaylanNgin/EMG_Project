import os
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Automatically find exactly where this script is saved on your Mac
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the exact path: Go to script folder -> step back once -> go to data -> find txt
INPUT_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data', 'emg_data.txt'))
# Sampling rate for converting samples -> seconds (used only if timestamps missing)
SAMPLE_RATE = 1000  # Hz
# ---------------------

# This will print the exact path Python is looking at so you can verify it
print(f"Looking for data at: {INPUT_FILE}")

try:
        # Try reading CSV of timestamp,value pairs; fallback to legacy single-column
        times = []
        values = []
        with open(INPUT_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) == 1:
                    # legacy value-only files
                    try:
                        values.append(float(parts[0]))
                    except ValueError:
                        continue
                else:
                    try:
                        ts = float(parts[0])
                        val = float(parts[1])
                        times.append(ts)
                        values.append(val)
                    except ValueError:
                        # skip malformed lines
                        continue

        if len(values) == 0:
            print("The file was empty or contained no valid numbers.")
        else:
            times = np.array(times)
            values = np.array(values)

            # If timestamps were not present, build them from sample rate
            if times.size == 0:
                n = values.shape[0]
                times = np.arange(n) / SAMPLE_RATE
            else:
                # Convert absolute epoch timestamps to relative seconds from start
                times = times - times[0]

            n = values.shape[0]
            print(f"Successfully loaded {n} data points.")
            print("Generating plot using timestamps...")

            plt.figure(figsize=(12, 5))
            plt.plot(times, values, color='b', linewidth=1.0)

            plt.title('Raw EMG Signal')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (ADC Value)')
            plt.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.show()

except FileNotFoundError:
    print(f"\n[ERROR] Could not find the file.")
    print("Double-check that your text file is actually named 'emg_data.txt' and is inside the 'data' folder!")