import os
import sys
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np


def parse_emg_file(path):
    times = []
    values = []

    with open(path, 'r') as f:
        for line in f:
            text = line.strip()
            if not text:
                continue

            parts = text.split(',')
            if len(parts) == 1:
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
                    continue

    values = np.array(values, dtype=float)
    times = np.array(times, dtype=float)

    if values.size == 0:
        raise ValueError(f"No valid EMG samples found in '{path}'.")

    if times.size == 0:
        times = np.arange(values.shape[0]) / 1000.0
    else:
        times = times - times[0]

    return times, values


def plot_emg_files(file_paths):
    plt.style.use('seaborn-v0_8')
    if not file_paths:
        return

    # --- Separate files into hold and release/open lists ---
    hold_paths = sorted([p for p in file_paths if 'hold' in os.path.basename(p).lower()])
    release_paths = sorted([p for p in file_paths if 'open' in os.path.basename(p).lower()])

    # --- Pre-scan ALL files to find global Y-axis limits ---
    global_min, global_max = np.inf, -np.inf
    all_data = {}
    for path in file_paths:
        times, values = parse_emg_file(path)
        all_data[path] = (times, values)
        if values.size > 0:
            global_min = min(global_min, np.min(values))
            global_max = max(global_max, np.max(values))

    # Add a bit of padding to the y-axis
    y_range = global_max - global_min if np.isfinite(global_min) and np.isfinite(global_max) else 0
    y_padding = y_range * 0.05
    ylim = (global_min - y_padding, global_max + y_padding)

    # --- Create a 2-column plot grid ---
    num_rows = max(len(hold_paths), len(release_paths))
    if num_rows == 0:
        print("No 'hold' or 'open' trials found in selected files.")
        return

    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows), squeeze=False)

    # --- Plot Hold trials on the left column ---
    for i, path in enumerate(hold_paths):
        times, values = all_data[path]
        ax = axes[i, 0]
        ax.plot(times, values, linewidth=1.0, color='royalblue')
        ax.set_title(os.path.splitext(os.path.basename(path))[0], fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (ADC Value)')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim(ylim)

    # --- Plot Release trials on the right column ---
    for i, path in enumerate(release_paths):
        times, values = all_data[path]
        ax = axes[i, 1]
        ax.plot(times, values, linewidth=1.0, color='coral')
        ax.set_title(os.path.splitext(os.path.basename(path))[0], fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (ADC Value)')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim(ylim)

    # Set titles for columns
    if num_rows > 0:
        axes[0, 0].set_title(f'Hold Trials\n{os.path.splitext(os.path.basename(hold_paths[0]))[0]}', fontsize=10) if hold_paths else None
        axes[0, 1].set_title(f'Release/Open Trials\n{os.path.splitext(os.path.basename(release_paths[0]))[0]}', fontsize=10) if release_paths else None


    # Hide any unused axes
    for i in range(len(hold_paths), num_rows):
        axes[i, 0].set_visible(False)
    for i in range(len(release_paths), num_rows):
        axes[i, 1].set_visible(False)

    fig.suptitle('EMG Trial Viewer: Hold vs. Release', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def ask_for_files():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_paths = filedialog.askopenfilenames(
        title='Select EMG text files to plot',
        filetypes=[('Text Files', '*.txt'), ('All Files', '*.*')],
    )

    root.destroy()
    return list(file_paths)


def main():
    if len(sys.argv) > 1:
        file_paths = sys.argv[1:]
    else:
        file_paths = ask_for_files()

    if not file_paths:
        print('No files selected. Exiting.')
        return

    try:
        plot_emg_files(file_paths)
    except Exception as exc:
        print(f'Error while loading EMG files: {exc}')


if __name__ == '__main__':
    main()
