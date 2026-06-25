import serial
import time
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import os
import numpy as np

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/cu.usbmodem3103'
BAUD_RATE = 200000
# Set to None to record indefinitely until user stops with Ctrl+C
DURATION = None
SAMPLE_RATE = 1000  # Hz (samples per second)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data', 'emg_data.txt'))
# Size of the visible plotting window in seconds
PLOT_WINDOW_SECONDS = 5
MIN_WINDOW_SECONDS = 0.5
MAX_WINDOW_SECONDS = 60.0
ZOOM_FACTOR = 1.5

# ---------------------

# Shared containers for the full capture session
data = []
timestamps = []

# Thread communication
plot_queue = queue.Queue()
writer_queue = queue.Queue()
stop_event = threading.Event()
pause_event = threading.Event()
data_lock = threading.Lock()
reader_thread = None
writer_thread = None

def parse_sample(line):
    """Accept either raw values or timestamp,value lines."""
    parts = line.strip().split(',')
    if not parts or not parts[-1]:
        return None

    try:
        return int(float(parts[-1]))
    except ValueError:
        return None


def serial_reader(port, baud, plot_q, writer_q, stop_evt, pause_evt):
    buf = ""
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
    except Exception as e:
        print(f"[ERROR] Could not open serial port: {e}")
        stop_evt.set()
        return

    with ser:
        while not stop_evt.is_set():
            if pause_evt.is_set():
                buf = ""
                ser.reset_input_buffer()
                time.sleep(0.05)
                continue

            try:
                n = ser.in_waiting
                if n > 0:
                    raw = ser.read(n).decode('utf-8', errors='ignore')
                    buf += raw
                    if '\n' in buf:
                        parts = buf.split('\n')
                        buf = parts.pop()
                        for line in parts:
                            s = line.strip()
                            if not s:
                                continue
                            val = parse_sample(s)
                            if val is None:
                                continue

                            ts = time.time()
                            sample = (ts, val)
                            plot_q.put(sample)
                            writer_q.put(sample)
                else:
                    time.sleep(0.001)
            except Exception:
                # nonfatal, keep running
                time.sleep(0.01)

def file_writer(q, out_path, stop_evt, flush_interval=0.5, batch_size=200):
    try:
        # Write CSV with timestamp,value so offline plotting can use real timing
        with open(out_path, 'w') as f:
            buffer = []
            last_flush = time.time()
            while not stop_evt.is_set() or not q.empty():
                try:
                    item = q.get(timeout=0.1)
                    buffer.append(item)
                    if len(buffer) >= batch_size:
                        for ts, val in buffer:
                            f.write(f"{ts},{val}\n")
                        buffer.clear()
                        f.flush()
                        last_flush = time.time()
                except queue.Empty:
                    pass

                if buffer and (time.time() - last_flush) > flush_interval:
                    for ts, val in buffer:
                        f.write(f"{ts},{val}\n")
                    buffer.clear()
                    f.flush()
                    last_flush = time.time()
            # flush any remaining
            for ts, val in buffer:
                f.write(f"{ts},{val}\n")
            f.flush()
    except Exception as e:
        print(f"[ERROR] file_writer: {e}")

start_time = None
pause_started_at = None
paused_total = 0.0
samples_received = 0
last_status_time = time.time()
is_started = False
is_paused = False
is_updating_slider = False
view_window_seconds = PLOT_WINDOW_SECONDS

# --- PLOT SETUP ---
fig, ax = plt.subplots(figsize=(12, 5))
fig.subplots_adjust(bottom=0.30)
line, = ax.plot([], [], color='b', linewidth=1.0, label='Channel 1')
ax.set_ylim(0, 4095)
ax.set_xlim(0, view_window_seconds)
if DURATION:
    ax.set_title(f'Live EMG Stream (Recording for {DURATION}s)')
else:
    ax.set_title('Live EMG Stream (Recording - press Ctrl+C to stop)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (ADC Value)')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper right')

start_ax = fig.add_axes([0.20, 0.16, 0.12, 0.06])
pause_ax = fig.add_axes([0.34, 0.16, 0.12, 0.06])
zoom_in_ax = fig.add_axes([0.48, 0.16, 0.12, 0.06])
zoom_out_ax = fig.add_axes([0.62, 0.16, 0.12, 0.06])
end_ax = fig.add_axes([0.76, 0.16, 0.12, 0.06])
slider_ax = fig.add_axes([0.12, 0.075, 0.78, 0.03])

start_button = Button(start_ax, 'Start')
pause_button = Button(pause_ax, 'Pause')
zoom_in_button = Button(zoom_in_ax, 'Zoom In')
zoom_out_button = Button(zoom_out_ax, 'Zoom Out')
end_button = Button(end_ax, 'End')
time_slider = Slider(slider_ax, 'Time', 0.0, view_window_seconds, valinit=0.0)

def init():
    line.set_data([], [])
    return line,


def get_capture_snapshot():
    with data_lock:
        xs = np.array(timestamps, dtype=float)
        ys = np.array(data, dtype=float)

    return xs, ys


def set_slider_range(total_time):
    slider_max = max(view_window_seconds, total_time)

    if time_slider.valmax != slider_max:
        time_slider.valmax = slider_max
        time_slider.ax.set_xlim(time_slider.valmin, slider_max)


def set_slider_value(value):
    global is_updating_slider

    value = max(time_slider.valmin, min(time_slider.valmax, value))
    if abs(time_slider.val - value) < 0.001:
        return

    is_updating_slider = True
    time_slider.set_val(value)
    is_updating_slider = False


def clear_queue(q):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


def get_active_elapsed(now=None):
    if start_time is None:
        return 0.0

    if now is None:
        now = time.time()

    active_paused_total = paused_total
    if is_paused and pause_started_at is not None:
        active_paused_total += now - pause_started_at

    return max(0.0, now - start_time - active_paused_total)


def render_window(view_start, update_slider_range=True):
    xs, ys = get_capture_snapshot()
    if xs.size == 0 or ys.size == 0:
        ax.set_xlim(0, view_window_seconds)
        return

    total_time = xs[-1]
    if update_slider_range:
        set_slider_range(total_time)

    max_start = max(0.0, total_time - view_window_seconds)
    view_start = max(0.0, min(view_start, max_start))
    view_end = view_start + view_window_seconds

    start_idx = np.searchsorted(xs, view_start, side='left')
    end_idx = np.searchsorted(xs, view_end, side='right')
    plot_x = xs[start_idx:end_idx]
    window_y = ys[start_idx:end_idx]

    if plot_x.size == 0 or window_y.size == 0:
        line.set_data([], [])
        ax.set_xlim(view_start, view_end)
        return

    # --- EMG SIGNAL PROCESSING ---
    baseline = np.mean(window_y)
    centered = window_y - baseline
    rectified = np.abs(centered)
    smooth_window = 150  # 150 samples = 150ms smoothing window
    if rectified.size >= smooth_window:
        plot_y = np.convolve(rectified, np.ones(smooth_window) / smooth_window, mode='same')
    else:
        plot_y = rectified
    # --------------------------------

    step = 5
    line.set_data(plot_x[::step], plot_y[::step])
    ax.set_xlim(view_start, view_end)
    ax.set_ylim(0, 4095)


def start_capture(event):
    global reader_thread, writer_thread, start_time, pause_started_at, paused_total, last_status_time
    global is_started, is_paused, samples_received
    global plot_queue, writer_queue, stop_event, pause_event

    if is_started:
        return

    print(f"Connecting to {SERIAL_PORT}...")
    plot_queue = queue.Queue()
    writer_queue = queue.Queue()
    stop_event = threading.Event()
    pause_event = threading.Event()

    with data_lock:
        timestamps.clear()
        data.clear()

    samples_received = 0
    start_time = time.time()
    pause_started_at = None
    paused_total = 0.0
    last_status_time = start_time
    is_started = True
    is_paused = False

    set_slider_value(0.0)
    line.set_data([], [])
    ax.set_xlim(0, view_window_seconds)
    ax.set_ylim(0, 4095)
    start_button.label.set_text('Started')
    pause_button.label.set_text('Pause')

    reader_thread = threading.Thread(
        target=serial_reader,
        args=(SERIAL_PORT, BAUD_RATE, plot_queue, writer_queue, stop_event, pause_event),
        daemon=True,
    )
    writer_thread = threading.Thread(
        target=file_writer,
        args=(writer_queue, OUTPUT_FILE, stop_event),
        daemon=True,
    )
    reader_thread.start()
    writer_thread.start()
    print("Connected! Starting live capture (press Ctrl+C to stop)...")


def toggle_pause(event):
    global is_paused, pause_started_at, paused_total

    if not is_started:
        return

    is_paused = not is_paused
    if is_paused:
        pause_started_at = time.time()
        pause_event.set()
        clear_queue(plot_queue)
        clear_queue(writer_queue)
    else:
        if pause_started_at is not None:
            paused_total += time.time() - pause_started_at
            pause_started_at = None
        clear_queue(plot_queue)
        clear_queue(writer_queue)
        pause_event.clear()
    pause_button.label.set_text('Resume' if is_paused else 'Pause')


def scrub_time(value):
    global is_paused, pause_started_at

    if is_updating_slider or not is_started:
        return

    if not is_paused:
        pause_started_at = time.time()
    is_paused = True
    pause_event.set()
    clear_queue(plot_queue)
    clear_queue(writer_queue)
    pause_button.label.set_text('Resume')
    view_start = max(0.0, value - view_window_seconds)
    render_window(view_start, update_slider_range=False)


def change_zoom(multiplier):
    global view_window_seconds

    view_window_seconds = max(
        MIN_WINDOW_SECONDS,
        min(MAX_WINDOW_SECONDS, view_window_seconds * multiplier),
    )

    xs, _ = get_capture_snapshot()
    if xs.size == 0:
        set_slider_range(0.0)
        ax.set_xlim(0, view_window_seconds)
        return

    total_time = xs[-1]
    if is_paused:
        set_slider_range(total_time)
        view_end = max(0.0, min(time_slider.val, total_time))
        view_start = max(0.0, view_end - view_window_seconds)
        render_window(view_start, update_slider_range=False)
        set_slider_value(view_end)
    else:
        set_slider_range(total_time)
        view_start = max(0.0, total_time - view_window_seconds)
        render_window(view_start)
        set_slider_value(total_time)


def zoom_in(event):
    change_zoom(1.0 / ZOOM_FACTOR)


def zoom_out(event):
    change_zoom(ZOOM_FACTOR)


def end_capture(event):
    stop_event.set()
    plt.close(fig)


start_button.on_clicked(start_capture)
pause_button.on_clicked(toggle_pause)
zoom_in_button.on_clicked(zoom_in)
zoom_out_button.on_clicked(zoom_out)
end_button.on_clicked(end_capture)
time_slider.on_changed(scrub_time)


def update_from_queue():
    global samples_received, last_status_time

    if not is_started:
        return True

    if DURATION is not None and get_active_elapsed() > DURATION:
        stop_event.set()
        try:
            plt.close(fig)
        except Exception:
            pass
        return False

    if is_paused:
        view_start = max(0.0, time_slider.val - view_window_seconds)
        render_window(view_start, update_slider_range=False)
        return True

    # 1. Drain the queue completely
    drained = 0
    while not plot_queue.empty():
        try:
            ts, val = plot_queue.get_nowait()
            with data_lock:
                timestamps.append(get_active_elapsed(ts))
                data.append(val)
            drained += 1
        except queue.Empty:
            break
    samples_received += drained

    now = time.time()
    if now - last_status_time >= 1.0:
        print(f"\rSamples received: {samples_received:,}", end="", flush=True)
        last_status_time = now

    with data_lock:
        xs = list(timestamps)
        ys = list(data)

    if not xs or not ys:
        return True

    total_time = xs[-1]

    set_slider_range(total_time)
    live_start = max(0.0, total_time - view_window_seconds)
    render_window(live_start)
    set_slider_value(total_time)

    return True

# --- WINDOW EXECUTION LOOP ---
try:
    print("Starting live visualization. Click Start to begin capture.")
    
    plt.ion() 
    plt.show(block=False) 

    while plt.fignum_exists(fig.number) and not stop_event.is_set():
        alive = update_from_queue()
        if not alive:
            break
        
        # A much faster, non-blocking render loop compared to plt.pause()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        
        # Cap the frame rate at ~30 FPS so the CPU can breathe
        time.sleep(0.03) 

except KeyboardInterrupt:
    print("\nForce stopped by user.")
except Exception as e:
    print(f"[ERROR] Loop error: {e}")
finally:
    stop_event.set()
    if reader_thread is not None:
        reader_thread.join(timeout=1.0)
    if writer_thread is not None:
        writer_thread.join(timeout=1.0)
    print(f"Hardware disconnected. Data saved to:\n{os.path.abspath(OUTPUT_FILE)}")
