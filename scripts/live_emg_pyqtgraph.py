import os
import queue
import sys
import threading
import time

import numpy as np
import pyqtgraph as pg
import serial
from pyqtgraph.Qt import QtCore, QtWidgets

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/cu.usbmodem3103'
BAUD_RATE = 200000
# Set to None to record indefinitely until user stops with End/Ctrl+C
DURATION = None
SAMPLE_RATE = 1000  # Hz (samples per second)
ADC_MIN = 0
ADC_MAX = 4095

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data', 'emg_data.txt'))
PLOT_WINDOW_SECONDS = 5.0
MIN_WINDOW_SECONDS = 0.5
MAX_WINDOW_SECONDS = 60.0
ZOOM_FACTOR = 1.5
SLIDER_SCALE = 1000
PLOT_PROCESSED_SIGNAL = False

if hasattr(QtCore.Qt, 'Orientation'):
    HORIZONTAL = QtCore.Qt.Orientation.Horizontal
else:
    HORIZONTAL = QtCore.Qt.Horizontal


def parse_sample(line):
    """Accept either raw values or timestamp,value lines."""
    parts = line.strip().split(',')
    if not parts or not parts[-1]:
        return None

    try:
        value = int(float(parts[-1]))
    except ValueError:
        return None

    if value < ADC_MIN or value > ADC_MAX:
        return None

    return value


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

                            sample = (time.time(), val)
                            plot_q.put(sample)
                            writer_q.put(sample)
                else:
                    time.sleep(0.001)
            except Exception:
                time.sleep(0.01)


def file_writer(q, out_path, stop_evt, flush_interval=0.5, batch_size=200):
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
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

            for ts, val in buffer:
                f.write(f"{ts},{val}\n")
            f.flush()
    except Exception as e:
        print(f"[ERROR] file_writer: {e}")


def clear_queue(q):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


class EmgWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Live EMG Stream - PyQtGraph')
        self.resize(1200, 650)

        self.data = []
        self.timestamps = []
        self.data_lock = threading.Lock()
        self.plot_queue = queue.Queue()
        self.writer_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.reader_thread = None
        self.writer_thread = None

        self.start_time = None
        self.pause_started_at = None
        self.paused_total = 0.0
        self.samples_received = 0
        self.last_status_time = time.time()
        self.is_started = False
        self.is_paused = False
        self.is_updating_slider = False
        self.view_window_seconds = PLOT_WINDOW_SECONDS

        self._build_ui()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_from_queue)
        self.timer.start(30)

    def _build_ui(self):
        pg.setConfigOptions(antialias=False)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setLabel('left', 'Amplitude (ADC Value)')
        self.plot_widget.getAxis('left').enableAutoSIPrefix(False)
        self.plot_widget.setYRange(ADC_MIN, ADC_MAX, padding=0)
        self.plot_widget.setXRange(0, self.view_window_seconds, padding=0)
        self.plot_widget.enableAutoRange(axis='y', enable=False)
        self.plot_widget.setLimits(yMin=ADC_MIN, yMax=ADC_MAX)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.35)
        self.curve = self.plot_widget.plot([], [], pen=pg.mkPen('#00a2ff', width=2))
        layout.addWidget(self.plot_widget, stretch=1)

        controls = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton('Start')
        self.pause_button = QtWidgets.QPushButton('Pause')
        self.zoom_in_button = QtWidgets.QPushButton('Zoom In')
        self.zoom_out_button = QtWidgets.QPushButton('Zoom Out')
        self.end_button = QtWidgets.QPushButton('End')

        controls.addWidget(self.start_button)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.zoom_in_button)
        controls.addWidget(self.zoom_out_button)
        controls.addWidget(self.end_button)
        layout.addLayout(controls)

        slider_row = QtWidgets.QHBoxLayout()
        slider_row.addWidget(QtWidgets.QLabel('Time'))
        self.time_slider = QtWidgets.QSlider(HORIZONTAL)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(int(self.view_window_seconds * SLIDER_SCALE))
        self.time_slider.setValue(0)
        slider_row.addWidget(self.time_slider)
        layout.addLayout(slider_row)

        self.status_label = QtWidgets.QLabel('Click Start to begin capture.')
        layout.addWidget(self.status_label)

        self.setCentralWidget(central)

        self.start_button.clicked.connect(self.start_capture)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.zoom_in_button.clicked.connect(lambda: self.change_zoom(1.0 / ZOOM_FACTOR))
        self.zoom_out_button.clicked.connect(lambda: self.change_zoom(ZOOM_FACTOR))
        self.end_button.clicked.connect(self.close)
        self.time_slider.valueChanged.connect(self.scrub_time)

    def get_capture_snapshot(self):
        with self.data_lock:
            xs = np.array(self.timestamps, dtype=float)
            ys = np.array(self.data, dtype=float)
        return xs, ys

    def set_slider_range(self, total_time):
        slider_max = max(self.view_window_seconds, total_time)
        self.time_slider.setMaximum(int(slider_max * SLIDER_SCALE))

    def set_slider_value(self, value):
        value = max(0.0, min(self.time_slider.maximum() / SLIDER_SCALE, value))
        slider_value = int(value * SLIDER_SCALE)
        if abs(self.time_slider.value() - slider_value) < 1:
            return

        self.is_updating_slider = True
        self.time_slider.setValue(slider_value)
        self.is_updating_slider = False

    def get_active_elapsed(self, now=None):
        if self.start_time is None:
            return 0.0

        if now is None:
            now = time.time()

        active_paused_total = self.paused_total
        if self.is_paused and self.pause_started_at is not None:
            active_paused_total += now - self.pause_started_at

        return max(0.0, now - self.start_time - active_paused_total)

    def render_window(self, view_start, update_slider_range=True):
        xs, ys = self.get_capture_snapshot()
        if xs.size == 0 or ys.size == 0:
            self.curve.setData([], [])
            self.plot_widget.setXRange(0, self.view_window_seconds, padding=0)
            self.plot_widget.setYRange(ADC_MIN, ADC_MAX, padding=0)
            return

        total_time = xs[-1]
        if update_slider_range:
            self.set_slider_range(total_time)

        max_start = max(0.0, total_time - self.view_window_seconds)
        view_start = max(0.0, min(view_start, max_start))
        view_end = view_start + self.view_window_seconds

        start_idx = np.searchsorted(xs, view_start, side='left')
        end_idx = np.searchsorted(xs, view_end, side='right')
        plot_x = xs[start_idx:end_idx]
        window_y = ys[start_idx:end_idx]

        if plot_x.size == 0 or window_y.size == 0:
            self.curve.setData([], [])
            self.plot_widget.setXRange(view_start, view_end, padding=0)
            self.plot_widget.setYRange(ADC_MIN, ADC_MAX, padding=0)
            return

        if PLOT_PROCESSED_SIGNAL:
            baseline = np.mean(window_y)
            centered = window_y - baseline
            rectified = np.abs(centered)
            smooth_window = 150
            if rectified.size >= smooth_window:
                plot_y = np.convolve(rectified, np.ones(smooth_window) / smooth_window, mode='same')
            else:
                plot_y = rectified
        else:
            plot_y = window_y

        step = 2
        self.curve.setData(plot_x[::step], plot_y[::step])
        self.plot_widget.setXRange(view_start, view_end, padding=0)
        self.plot_widget.setYRange(ADC_MIN, ADC_MAX, padding=0)

    def start_capture(self):
        if self.is_started:
            return

        print(f"Connecting to {SERIAL_PORT}...")
        self.plot_queue = queue.Queue()
        self.writer_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

        with self.data_lock:
            self.timestamps.clear()
            self.data.clear()

        self.samples_received = 0
        self.start_time = time.time()
        self.pause_started_at = None
        self.paused_total = 0.0
        self.last_status_time = self.start_time
        self.is_started = True
        self.is_paused = False

        self.set_slider_value(0.0)
        self.curve.setData([], [])
        self.plot_widget.setXRange(0, self.view_window_seconds, padding=0)
        self.plot_widget.setYRange(ADC_MIN, ADC_MAX, padding=0)
        self.start_button.setText('Started')
        self.pause_button.setText('Pause')
        self.status_label.setText('Connected. Capturing live EMG data...')

        self.reader_thread = threading.Thread(
            target=serial_reader,
            args=(
                SERIAL_PORT,
                BAUD_RATE,
                self.plot_queue,
                self.writer_queue,
                self.stop_event,
                self.pause_event,
            ),
            daemon=True,
        )
        self.writer_thread = threading.Thread(
            target=file_writer,
            args=(self.writer_queue, OUTPUT_FILE, self.stop_event),
            daemon=True,
        )
        self.reader_thread.start()
        self.writer_thread.start()

    def toggle_pause(self):
        if not self.is_started:
            return

        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_started_at = time.time()
            self.pause_event.set()
            clear_queue(self.plot_queue)
            clear_queue(self.writer_queue)
        else:
            if self.pause_started_at is not None:
                self.paused_total += time.time() - self.pause_started_at
                self.pause_started_at = None
            clear_queue(self.plot_queue)
            clear_queue(self.writer_queue)
            self.pause_event.clear()

        self.pause_button.setText('Resume' if self.is_paused else 'Pause')

    def scrub_time(self, slider_value):
        if self.is_updating_slider or not self.is_started:
            return

        if not self.is_paused:
            self.pause_started_at = time.time()
        self.is_paused = True
        self.pause_event.set()
        clear_queue(self.plot_queue)
        clear_queue(self.writer_queue)
        self.pause_button.setText('Resume')

        value = slider_value / SLIDER_SCALE
        view_start = max(0.0, value - self.view_window_seconds)
        self.render_window(view_start, update_slider_range=False)

    def change_zoom(self, multiplier):
        self.view_window_seconds = max(
            MIN_WINDOW_SECONDS,
            min(MAX_WINDOW_SECONDS, self.view_window_seconds * multiplier),
        )

        xs, _ = self.get_capture_snapshot()
        if xs.size == 0:
            self.set_slider_range(0.0)
            self.plot_widget.setXRange(0, self.view_window_seconds, padding=0)
            return

        total_time = xs[-1]
        self.set_slider_range(total_time)
        if self.is_paused:
            view_end = max(0.0, min(self.time_slider.value() / SLIDER_SCALE, total_time))
        else:
            view_end = total_time

        view_start = max(0.0, view_end - self.view_window_seconds)
        self.render_window(view_start, update_slider_range=False)
        self.set_slider_value(view_end)

    def update_from_queue(self):
        if not self.is_started:
            return

        if DURATION is not None and self.get_active_elapsed() > DURATION:
            self.close()
            return

        if self.is_paused:
            view_start = max(0.0, self.time_slider.value() / SLIDER_SCALE - self.view_window_seconds)
            self.render_window(view_start, update_slider_range=False)
            return

        drained = 0
        while not self.plot_queue.empty():
            try:
                ts, val = self.plot_queue.get_nowait()
                with self.data_lock:
                    self.timestamps.append(self.get_active_elapsed(ts))
                    self.data.append(val)
                drained += 1
            except queue.Empty:
                break

        self.samples_received += drained

        now = time.time()
        if now - self.last_status_time >= 1.0:
            self.status_label.setText(f"Samples received: {self.samples_received:,}")
            print(f"\rSamples received: {self.samples_received:,}", end="", flush=True)
            self.last_status_time = now

        xs, _ = self.get_capture_snapshot()
        if xs.size == 0:
            return

        total_time = xs[-1]
        live_start = max(0.0, total_time - self.view_window_seconds)
        self.render_window(live_start)
        self.set_slider_value(total_time)

    def closeEvent(self, event):
        self.stop_event.set()
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=1.0)
        if self.writer_thread is not None:
            self.writer_thread.join(timeout=1.0)

        # Prompt user to choose a filename so trials don't overwrite each other.
        try:
            default_dir = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))
            os.makedirs(default_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            suggested = f"emg_data_{ts}.txt"
            default_path = os.path.join(default_dir, suggested)

            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save EMG Data As",
                default_path,
                "Text Files (*.txt);;All Files (*)",
            )

            if save_path:
                try:
                    if os.path.exists(OUTPUT_FILE):
                        os.replace(OUTPUT_FILE, save_path)
                        final_path = save_path
                    else:
                        final_path = save_path
                except Exception as e:
                    print(f"[ERROR] Could not move file: {e}")
                    final_path = os.path.abspath(OUTPUT_FILE)
            else:
                # User canceled the dialog - avoid silent overwrite by timestamping the file.
                if os.path.exists(OUTPUT_FILE):
                    fallback = os.path.join(default_dir, suggested)
                    try:
                        os.replace(OUTPUT_FILE, fallback)
                        final_path = fallback
                    except Exception:
                        final_path = os.path.abspath(OUTPUT_FILE)
                else:
                    final_path = os.path.abspath(OUTPUT_FILE)
        except Exception as e:
            print(f"[ERROR] save dialog: {e}")
            final_path = os.path.abspath(OUTPUT_FILE)

        print(f"\nHardware disconnected. Data saved to:\n{final_path}")
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = EmgWindow()
    window.show()
    if hasattr(app, 'exec'):
        sys.exit(app.exec())
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
