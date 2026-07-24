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

## Raspberry Pi: Subject 2 LDA

Clone the `new-features` branch and install the LDA-only dependencies:

```bash
git clone --branch new-features https://github.com/jaylanNgin/EMG_Project.git
cd EMG_Project
chmod +x scripts/setup_raspberry_pi.sh scripts/run_subject2_lda.sh
./scripts/setup_raspberry_pi.sh
source .venv/bin/activate
```

Re-run the complete clip-size comparison:

```bash
python scripts/sweep_lda_clip_sizes.py Subject2GloveOff
python scripts/sweep_lda_clip_sizes.py Subject2GloveOn
```

Run LDA with the best clip sizes found in the comparison:

```bash
./scripts/run_subject2_lda.sh off  # ResultClipSizeUp200
./scripts/run_subject2_lda.sh on   # ResultClipSizeUp300
```

The runner defaults to Matplotlib's headless backend so it works over SSH. To
show the confusion-matrix window on a Pi desktop, run:

```bash
MPLBACKEND=TkAgg ./scripts/run_subject2_lda.sh on
```
