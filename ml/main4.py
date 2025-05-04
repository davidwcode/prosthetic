import threading
import time
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import EMGDataset
from util import record_emg
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
import torchmetrics
import tkinter as tk
from tkinter import font, ttk
from model import HDClassifier
import serial

# Only 4 classes and even faster calibration

NUM_SAMPLES = 1000
NUM_CHANNELS = 4
SAMPLES_PER_POINT = 50
BATCH_SIZE = 1
STATE_DICT = {0: "Relax", 1: "Clench", 2: "Triton", 3: "L(ove)"}
SAMPLING_RATE = 500
DIMENSIONS = 10000
GENERAL_MODEL_PATH = "general_model.pt"
CALIB_DIR = "calibration_data"

# Create directories for calibration data
os.makedirs(CALIB_DIR, exist_ok=True)
for cls_name in STATE_DICT.values():
    os.makedirs(os.path.join(CALIB_DIR, cls_name), exist_ok=True)

class VaderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vader Hand Control")
        try:
            self.root.state('zoomed')
            self.root.attributes('-zoomed', True)
        except:
            pass
        self.root.configure(bg='black')

        self.stop_event = threading.Event()
        self.hardware_initialized = False
        self.continue_event = threading.Event()

        self.board_shim = None
        self.ser = None
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        # Models
        self.model = HDClassifier(DIMENSIONS, len(STATE_DICT), NUM_CHANNELS).to(self.device)
        self.general_model = HDClassifier(DIMENSIONS, len(STATE_DICT), NUM_CHANNELS).to(self.device)
        self._load_general_model()

        # Fonts/colors
        self.title_font = font.Font(size=32, weight='bold')
        self.button_font = font.Font(size=16)
        self.primary_color = 'red'
        self.accent_color = 'red'

        # Display
        self.display = tk.Label(
            root, text="Vader Control Center",
            fg=self.primary_color, bg='black', font=self.title_font,
            justify='center', wraplength=self.root.winfo_screenwidth()-100
        )
        self.display.pack(expand=True, fill='both', pady=20)

        # Progress bar
        style = ttk.Style()
        style.theme_use('classic')
        style.configure('Red.Horizontal.TProgressbar', troughcolor='black', background='red')
        self.progress = ttk.Progressbar(
            root, style='Red.Horizontal.TProgressbar', orient='horizontal', mode='determinate'
        )
        self.progress.pack(fill='x', padx=50, pady=10)

        # Buttons
        btn_frame = tk.Frame(root, bg='black')
        btn_frame.pack(pady=10)
        self.start_btn = tk.Button(
            btn_frame, text="Start Calibration",
            fg='black', bg=self.accent_color, font=self.button_font,
            width=16, command=self.start_process
        )
        self.start_btn.pack(side='left', expand=True, padx=5)
        self.general_btn = tk.Button(
            btn_frame, text="Use General Model",
            fg='black', bg=self.accent_color, font=self.button_font,
            width=16, command=self.start_general_inference
        )
        self.general_btn.pack(side='left', expand=True, padx=5)
        self.aggregate_btn = tk.Button(
            btn_frame, text="Aggregate",
            fg='black', bg=self.accent_color, font=self.button_font,
            width=16, command=self.start_aggregate
        )
        self.aggregate_btn.pack(side='left', expand=True, padx=5)
        self.aggregate_infer_btn = tk.Button(
            btn_frame, text="Infer Aggregate Model",
            fg='black', bg=self.accent_color, font=self.button_font,
            width=16, command=self.start_aggregate_inference
        )
        self.aggregate_infer_btn.pack(side='left', expand=True, padx=5)
        self.stop_btn = tk.Button(
            btn_frame, text="Stop",
            fg='black', bg=self.accent_color, font=self.button_font,
            width=16, state='disabled', command=self.stop_process
        )
        self.stop_btn.pack(side='left', expand=True, padx=5)

        self.continue_btn = tk.Button(
            root, text="Continue",
            fg='black', bg=self.accent_color, font=self.button_font,
            width=16, state='disabled', command=self._on_continue
        )
        self.continue_btn.pack(side='left', expand=True, padx=5)
    
    def _on_continue(self):
        self.continue_btn.pack_forget()
        self.continue_event.set()

    def _load_general_model(self):
        if os.path.exists(GENERAL_MODEL_PATH):
            try:
                self.general_model.load_state_dict(
                    torch.load(GENERAL_MODEL_PATH, map_location=self.device)
                )
            except Exception:
                pass

    def _save_general_model(self):
        torch.save(self.general_model.state_dict(), GENERAL_MODEL_PATH)

    def _save_window(self, cls_index, window):
        class_name = STATE_DICT[cls_index]
        cls_dir = os.path.join(CALIB_DIR, class_name)
        existing = len([f for f in os.listdir(cls_dir) if f.endswith('.npy')])
        filename = os.path.join(cls_dir, f"window_{existing}.npy")
        np.save(filename, window)

    def init_hardware(self):
        if not self.hardware_initialized:
            board_id = BoardIds.MINDROVE_WIFI_BOARD
            params = MindRoveInputParams()
            self.board_shim = BoardShim(board_id, params)
            self.board_shim.prepare_session()
            self.board_shim.start_stream(450000)
            self.ser = serial.Serial("/dev/cu.usbmodem11301", 9600)
            time.sleep(2)
            start = time.time()
            while time.time() - start < 10 and not self.stop_event.is_set():
                if self.board_shim.get_board_data_count() > 0:
                    self.board_shim.get_board_data(SAMPLING_RATE)
                    break
            self.hardware_initialized = True

    def start_process(self):
        self.start_btn.config(state='disabled')
        self.general_btn.config(state='disabled')
        self.aggregate_btn.config(state='disabled')
        self.aggregate_infer_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.stop_event.clear()
        threading.Thread(target=self.run_sequence, daemon=True).start()

    def start_general_inference(self):
        self.start_btn.config(state='disabled')
        self.general_btn.config(state='disabled')
        self.aggregate_btn.config(state='disabled')
        self.aggregate_infer_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.stop_event.clear()
        threading.Thread(target=self.run_general_inference, daemon=True).start()

    def start_aggregate(self):
        self.start_btn.config(state='disabled')
        self.general_btn.config(state='disabled')
        self.aggregate_btn.config(state='disabled')
        self.aggregate_infer_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.stop_event.clear()
        threading.Thread(target=self.run_aggregate, daemon=True).start()

    def start_aggregate_inference(self):
        self.start_btn.config(state='disabled')
        self.general_btn.config(state='disabled')
        self.aggregate_btn.config(state='disabled')
        self.aggregate_infer_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.stop_event.clear()
        threading.Thread(target=self.run_aggregate_inference, daemon=True).start()

    def stop_process(self):
        self.stop_event.set()

    def run_sequence(self):
        try:
            self.update_display("Initializing hardware...")
            self.init_hardware()
            num_classes = len(STATE_DICT)
            windows_per_class = NUM_SAMPLES // SAMPLES_PER_POINT
            total = num_classes * windows_per_class
            data = np.zeros((total, SAMPLES_PER_POINT, NUM_CHANNELS), np.float32)
            labels = np.zeros(total, np.int64)
            self.progress.config(maximum=windows_per_class)
            idx = 0
            with torch.no_grad():
                for cls in range(num_classes):
                    if self.stop_event.is_set(): return self.reset_ui()
                    state = STATE_DICT[cls]
                    self.ser.write(f"{cls}\n".encode())
                    for cnt in (3,2,1):
                        if self.stop_event.is_set(): return self.reset_ui()
                        self.update_display(f"{state} in {cnt}...")
                        time.sleep(1)
                    for w in range(windows_per_class):
                        if self.stop_event.is_set(): return self.reset_ui()
                        self.update_display(f"Collecting {state} ({w+1}/{windows_per_class})")
                        self.progress['value'] = w+1
                        rot, win = record_emg(self.board_shim, SAMPLES_PER_POINT, NUM_CHANNELS, SAMPLING_RATE)
                        self._save_window(cls, win)
                        data[idx] = win
                        labels[idx] = cls
                        idx += 1
                    self.progress['value'] = 0

            # Evaluate models
            self.update_display("Training/Evaluating...")
            time.sleep(5)
            X = torch.tensor(data, dtype=torch.float32)
            Y = torch.tensor(labels, dtype=torch.long)
            full_ds = EMGDataset(X, Y)
            train_len = int(len(full_ds)*0.8)
            train_ds, test_ds = torch.utils.data.random_split(full_ds, [train_len, len(full_ds)-train_len])
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
            self.model = HDClassifier(DIMENSIONS, num_classes, NUM_CHANNELS).to(self.device)
            self.general_model = HDClassifier(DIMENSIONS, num_classes, NUM_CHANNELS).to(self.device)
            self._load_general_model()
            with torch.no_grad():
                for xb,yb in train_loader:
                    xb,yb = xb.to(self.device), yb.to(self.device)
                    self.model.build(xb, yb)
                    self.general_model.build(xb, yb, lr=0.1)
            self.model.normalize()
            self._save_general_model
            results = {}
            for name, m in [("Custom", self.model), ("General", self.general_model)]:
                acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
                prec = torchmetrics.Precision(task="multiclass", average="none", num_classes=num_classes).to(self.device)
                rec = torchmetrics.Recall(task="multiclass", average="none", num_classes=num_classes).to(self.device)
                with torch.no_grad():
                    for xb,yb in test_loader:
                        xb,yb = xb.to(self.device), yb.to(self.device)
                        out = m(xb)
                        acc.update(out,yb)
                        prec.update(out,yb)
                        rec.update(out,yb)
                p, r = prec.compute().cpu().numpy()*100, rec.compute().cpu().numpy()*100
                results[name] = (acc.compute().item()*100, p, r)
            lines=[]
            for name,(a,p,r) in results.items():
                lines.append(f"{name} Model Acc: {a:.1f}%")
                for i in range(num_classes):
                    lines.append(f"  {STATE_DICT[i]} P:{p[i]:.1f}% R:{r[i]:.1f}%")
                lines.append("")
            
            self.update_display("\n".join(lines))
            # Wait for click on continue
            self.continue_btn.pack(side='left', expand=True, padx=5)
            self.root.after(0, lambda: self.continue_btn.config(state='normal'))
            self.continue_event.wait()
            self.continue_event.clear()
            time.sleep(2)
            self.run_custom_inference()
        except Exception as e:
            self.update_display(f"Error: {e}")
        finally:
            if self.board_shim:
                self.board_shim.stop_stream()
                self.board_shim.release_session()
            self.hardware_initialized = False
            self.reset_ui()

    def run_custom_inference(self, loop=True):
        try:
            if not self.hardware_initialized:
                self.update_display("Initializing hardware for custom inference...")
                self.init_hardware()
            self.update_display("Starting custom inference...")
            self.progress['value'] = 0
            while loop and not self.stop_event.is_set():
                self._inference_step(self.model)
        except Exception as e:
            self.update_display(f"Error: {e}")
        finally:
            if self.board_shim:
                self.board_shim.stop_stream()
                self.board_shim.release_session()
            self.hardware_initialized = False
            self.reset_ui()

    def run_general_inference(self):
        try:
            if not self.hardware_initialized:
                self.update_display("Initializing hardware for general inference...")
                self.init_hardware()
            model_copy = HDClassifier(DIMENSIONS, len(STATE_DICT), NUM_CHANNELS).to(self.device)
            model_copy.load_state_dict(self.general_model.state_dict())
            model_copy.normalize()
            self.update_display("Starting general inference...")
            self.progress['value'] = 0
            while not self.stop_event.is_set():
                self._inference_step(model_copy)
        except Exception as e:
            self.update_display(f"Error: {e}")
        finally:
            if self.board_shim:
                self.board_shim.stop_stream()
                self.board_shim.release_session()
            self.hardware_initialized = False
            self.reset_ui()

    def run_aggregate(self):
        try:
            self.update_display("Aggregating calibration data...")
            ds = load_calibration_dataset()
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
            agg_model = HDClassifier(DIMENSIONS, len(STATE_DICT), NUM_CHANNELS).to(self.device)
            with torch.no_grad():
                for X, Y in loader:
                    if self.stop_event.is_set(): break
                    X, Y = X.to(self.device), Y.to(self.device)
                    agg_model.build(X, Y)
            agg_model.normalize()
            self.aggregate_model = agg_model
            self.update_display(f"Aggregated model built ({len(loader.dataset)} samples)")
            time.sleep(2)
        except Exception as e:
            self.update_display(f"Error: {e}")
        finally:
            self.stop_event.set()
            self.reset_ui()

    def run_aggregate_inference(self):
        try:
            if not self.hardware_initialized:
                self.update_display("Initializing hardware for aggregate inference...")
                self.init_hardware()
            model_copy = HDClassifier(DIMENSIONS, len(STATE_DICT), NUM_CHANNELS).to(self.device)
            model_copy.load_state_dict(self.aggregate_model.state_dict())
            model_copy.normalize()
            self.update_display("Starting aggregate inference...")
            self.progress['value'] = 0
            while not self.stop_event.is_set():
                self._inference_step(model_copy)
        except Exception as e:
            self.update_display(f"Error: {e}")
        finally:
            if self.board_shim:
                self.board_shim.stop_stream()
                self.board_shim.release_session()
            self.hardware_initialized = False
            self.reset_ui()

    def _inference_step(self, model):
        num_classes = len(STATE_DICT)
        votes = np.zeros(num_classes, dtype=int)
        rotation = None
        for _ in range(10):
            if self.stop_event.is_set(): break
            rot, win = record_emg(self.board_shim, SAMPLES_PER_POINT, NUM_CHANNELS, SAMPLING_RATE)
            if rotation is None and rot != "None": rotation = rot
            xb = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(self.device)
            out = model(xb)
            votes[out.argmax().item()] += 1
        cls = votes.argmax()
        action = STATE_DICT[cls]
        if rotation=='Left': code=8
        elif rotation=='Right': code=9
        else: code=None
        if code:
            self.ser.write(f"{code}\n".encode())
        else:
            self.ser.write(f"{cls}\n".encode())
        self.update_display(f"Pose: {action}\nRotation: {rotation or 'None'}")

    def update_display(self, text):
        self.display.config(text=text)

    def reset_ui(self):
        self.start_btn.config(state='normal')
        self.general_btn.config(state='normal')
        self.aggregate_btn.config(state='normal')
        self.aggregate_infer_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.update_display("Welcome to Vader!")
        self.progress['value'] = 0
        self.stop_event.clear()

def load_calibration_dataset(calib_dir=CALIB_DIR):
    data_list = []
    labels = []
    for cls, name in STATE_DICT.items():
        cls_dir = os.path.join(calib_dir, name)
        if not os.path.exists(cls_dir):
            continue
        for fn in sorted(os.listdir(cls_dir)):
            if fn.endswith('.npy'):
                arr = np.load(os.path.join(cls_dir, fn))
                data_list.append(arr)
                labels.append(cls)
    if data_list:
        X = np.stack(data_list, axis=0)
        Y = np.array(labels, dtype=int)
        return EMGDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long))
    empty_X = np.zeros((0, SAMPLES_PER_POINT, NUM_CHANNELS), dtype=np.float32)
    empty_Y = np.zeros((0,), dtype=int)
    return EMGDataset(torch.tensor(empty_X, dtype=torch.float32), torch.tensor(empty_Y, dtype=torch.long))

if __name__ == '__main__':
    root = tk.Tk()
    app = VaderGUI(root)
    root.mainloop()
