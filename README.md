# EMG Project

This workspace contains simple scripts to record and visualize EMG data from an STM32 (USB serial).

Scripts:
- `scripts/record_emg.py` — simple recorder that writes raw ADC values to `data/emg_data.txt`.
- `scripts/live_emg.py` — improved live visualizer: threaded serial reader, buffered writer, and Matplotlib animation (blitting enabled).
- `scripts/plot_emg.py` — offline plot from `data/emg_data.txt` using NumPy + Matplotlib.

Quick start:

1. Install dependencies (prefer a virtualenv):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Confirm your serial device path (on macOS):

```bash
ls /dev/cu.*
```

3. Edit the `SERIAL_PORT` and `BAUD_RATE` constants at the top of the scripts if needed.

4. To run live visualization:

```bash
python3 scripts/live_emg.py
```

5. To record only:

```bash
python3 scripts/record_emg.py
```

6. To plot previously recorded data:

```bash
python3 scripts/plot_emg.py
```

7. To open one or more saved EMG `.txt` trials in a file picker and plot them together:

```bash
python3 viewers/view_emg_files.py
```

Or pass filenames directly:

```bash
python3 viewers/view_emg_files.py data/Trial1.txt data/Trial2.txt
```

Notes & tips
- If plotting at high sample rates is slow, consider installing and using `pyqtgraph` instead of Matplotlib for the live plot.
- If you see permission errors opening `/dev/cu.*`, try running with `sudo` or add your user to relevant groups (on macOS usually not necessary).
- If the serial port doesn't open, confirm your STM32 firmware serial settings and that no other application (CoolTerm, screen) is connected.
