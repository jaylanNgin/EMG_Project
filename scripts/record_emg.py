import serial
import time
import os

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/cu.usbmodem103'
BAUD_RATE = 200000  # Updated to match your new C code
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Use an absolute path for clarity
OUTPUT_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data', 'emg_data.txt'))
# If you want to keep previous captures set APPEND=True
APPEND = False
# ---------------------

mode = 'a' if APPEND else 'w'

try:
    print(f"Connecting to {SERIAL_PORT} at {BAUD_RATE} baud...")
    # use context manager for auto-close
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
        print("Connected! Waiting for data...")

        print(f"Recording 1000Hz EMG data to '{OUTPUT_FILE}'...")
        print("Press Ctrl+C to stop recording.")
        print("-" * 40)

        # Use a regular file handle and ensure it's closed on exit
        with open(OUTPUT_FILE, mode) as file:
            while True:
                if ser.in_waiting > 0:
                    raw_line = ser.readline()
                    try:
                        clean_data = raw_line.decode('utf-8').strip()
                        if clean_data:
                            print(clean_data)
                            file.write(clean_data + '\n')
                    except UnicodeDecodeError:
                        pass

except serial.SerialException as e:
    print(f"\n[ERROR] Could not connect to the port: {e}")
    print("Did you remember to close CoolTerm?")
except KeyboardInterrupt:
    print("\n\nRecording stopped by user.")
finally:
    print(f"Data saved to {OUTPUT_FILE}.")
