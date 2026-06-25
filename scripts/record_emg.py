import serial
import time
import os
from datetime import datetime

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/cu.usbmodem3103'
BAUD_RATE = 200000  # Updated to match your new C code
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))
# ---------------------


def safe_filename(name):
    cleaned = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in name.strip())
    return cleaned.strip('_')


def get_output_file():
    os.makedirs(DATA_DIR, exist_ok=True)

    default_name = f"emg_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    user_name = input(f"Save this recording as [{default_name}]: ").strip()
    base_name = safe_filename(user_name) if user_name else default_name

    if not base_name.lower().endswith('.txt'):
        base_name += '.txt'

    output_file = os.path.abspath(os.path.join(DATA_DIR, base_name))
    name_root, extension = os.path.splitext(output_file)
    counter = 2

    while os.path.exists(output_file):
        output_file = f"{name_root}_{counter}{extension}"
        counter += 1

    return output_file


OUTPUT_FILE = None

try:
    OUTPUT_FILE = get_output_file()

    print(f"Connecting to {SERIAL_PORT} at {BAUD_RATE} baud...")
    # use context manager for auto-close
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
        print("Connected! Waiting for data...")

        print(f"Recording 1000Hz EMG data to '{OUTPUT_FILE}'...")
        print("Press Ctrl+C to stop recording.")
        print("-" * 40)

        # Use a regular file handle and ensure it's closed on exit
        with open(OUTPUT_FILE, 'w') as file:
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
    if OUTPUT_FILE:
        print(f"Data saved to {OUTPUT_FILE}.")
