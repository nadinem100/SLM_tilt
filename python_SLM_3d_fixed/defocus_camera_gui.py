# -*- coding: utf-8 -*-
"""
Defocus Camera GUI
Select between BMP files with different defocus values,
load onto SLM, and view the result on the Basler camera.
"""

import sys
import os
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
from datetime import datetime

# Add paths for imports
HOLOEYE_PATH = r"C:\Users\srtwe\Box\EndresLab\z_Second Experiment\Code\Holoeye_SLM\examples - Copy"
sys.path.insert(0, HOLOEYE_PATH)
sys.path.insert(0, os.path.dirname(HOLOEYE_PATH))

# Import SLM SDK
try:
    import HEDS
    from hedslib.heds_types import *
    SLM_AVAILABLE = True
except ImportError:
    SLM_AVAILABLE = False
    print("Warning: HEDS (Holoeye SDK) not available")

# Import Basler camera
try:
    from camera_feedback.basler_camera import BaslerCamera
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("Warning: Basler camera not available")


class DefocusCameraGUI:
    def __init__(self, root, bmp_folder=None):
        self.root = root
        self.root.title("Defocus Camera Viewer")
        self.root.configure(bg='#1a1a2e')
        
        # Default folder
        if bmp_folder is None:
            bmp_folder = r"c:\Users\srtwe\Box\EndresLab\z_Second Experiment\Code\SLM simulation\nadine\DMD_SLM\slm_output_paraxial\adaptive_test_fixed\defocus_sweep"
        self.bmp_folder = Path(bmp_folder)
        
        # State
        self.slm = None
        self.slm_initialized = False
        self.camera = None
        self.camera_initialized = False
        self.current_bmp = None
        self.bmp_files = []
        
        # Camera settings
        self.exposure_us = 500  # Default exposure
        self.num_averages = 3
        
        # Setup GUI
        self.setup_gui()
        
        # Initialize hardware
        self.init_slm()
        self.init_camera()
        
        # Load BMP list
        self.refresh_bmp_list()
    
    def setup_gui(self):
        # Colors
        bg = '#1a1a2e'
        fg = '#eef1ff'
        accent = '#e94560'
        highlight = '#4ecca3'
        entry_bg = '#16213e'
        
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background=bg)
        style.configure('TLabel', background=bg, foreground=fg, font=('Consolas', 10))
        style.configure('Title.TLabel', background=bg, foreground=accent, font=('Consolas', 16, 'bold'))
        style.configure('Status.TLabel', background=bg, foreground=highlight, font=('Consolas', 9))
        
        # Main container
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(main, text="DEFOCUS CAMERA VIEWER", style='Title.TLabel').pack(pady=(0, 10))
        
        # === LEFT: BMP Selection ===
        left_frame = ttk.Frame(main)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Folder selection
        folder_frame = ttk.Frame(left_frame)
        folder_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(folder_frame, text="BMP Folder:").pack(anchor=tk.W)
        
        folder_row = ttk.Frame(folder_frame)
        folder_row.pack(fill=tk.X)
        
        self.folder_var = tk.StringVar(value=str(self.bmp_folder))
        folder_entry = tk.Entry(folder_row, textvariable=self.folder_var, 
                               font=('Consolas', 8), bg=entry_bg, fg=fg, width=40)
        folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        browse_btn = tk.Button(folder_row, text="...", command=self.browse_folder,
                              bg=accent, fg='white', font=('Consolas', 9, 'bold'),
                              relief=tk.FLAT, cursor='hand2')
        browse_btn.pack(side=tk.LEFT, padx=2)
        
        refresh_btn = tk.Button(folder_row, text="Refresh", command=self.refresh_bmp_list,
                               bg=entry_bg, fg=fg, font=('Consolas', 8),
                               relief=tk.FLAT, cursor='hand2')
        refresh_btn.pack(side=tk.LEFT, padx=2)
        
        # BMP Listbox
        ttk.Label(left_frame, text="Select BMP:").pack(anchor=tk.W, pady=(10, 2))
        
        listbox_frame = ttk.Frame(left_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True)
        
        self.bmp_listbox = tk.Listbox(listbox_frame, font=('Consolas', 9),
                                      bg=entry_bg, fg=fg, selectbackground=accent,
                                      width=45, height=15)
        self.bmp_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.bmp_listbox.bind('<<ListboxSelect>>', self.on_bmp_select)
        
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.bmp_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.bmp_listbox.config(yscrollcommand=scrollbar.set)
        
        # Load button
        self.load_btn = tk.Button(left_frame, text="LOAD ON SLM", command=self.load_on_slm,
                                 bg=highlight, fg='#1a1a2e', font=('Consolas', 11, 'bold'),
                                 relief=tk.FLAT, cursor='hand2', height=2)
        self.load_btn.pack(fill=tk.X, pady=(10, 5))
        
        # Camera controls
        cam_frame = ttk.Frame(left_frame)
        cam_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(cam_frame, text="Camera Controls:").pack(anchor=tk.W)
        
        exp_row = ttk.Frame(cam_frame)
        exp_row.pack(fill=tk.X, pady=2)
        ttk.Label(exp_row, text="Exposure (us):").pack(side=tk.LEFT)
        self.exp_var = tk.StringVar(value=str(self.exposure_us))
        exp_entry = tk.Entry(exp_row, textvariable=self.exp_var, font=('Consolas', 10),
                            bg=entry_bg, fg=fg, width=10)
        exp_entry.pack(side=tk.LEFT, padx=5)
        exp_entry.bind('<Return>', lambda e: self.update_exposure())
        
        avg_row = ttk.Frame(cam_frame)
        avg_row.pack(fill=tk.X, pady=2)
        ttk.Label(avg_row, text="Averages:").pack(side=tk.LEFT)
        self.avg_var = tk.StringVar(value=str(self.num_averages))
        avg_entry = tk.Entry(avg_row, textvariable=self.avg_var, font=('Consolas', 10),
                            bg=entry_bg, fg=fg, width=10)
        avg_entry.pack(side=tk.LEFT, padx=5)
        
        # Capture button
        self.capture_btn = tk.Button(left_frame, text="CAPTURE IMAGE", command=self.capture_image,
                                    bg=accent, fg='white', font=('Consolas', 11, 'bold'),
                                    relief=tk.FLAT, cursor='hand2', height=2)
        self.capture_btn.pack(fill=tk.X, pady=5)
        
        # Auto-capture checkbox
        self.auto_capture_var = tk.BooleanVar(value=True)
        auto_cb = tk.Checkbutton(left_frame, text="Auto-capture on load", 
                                variable=self.auto_capture_var,
                                bg=bg, fg=fg, selectcolor=entry_bg,
                                font=('Consolas', 9))
        auto_cb.pack(anchor=tk.W)
        
        # Save button
        save_btn = tk.Button(left_frame, text="Save Current Image", command=self.save_image,
                            bg=entry_bg, fg=fg, font=('Consolas', 9),
                            relief=tk.FLAT, cursor='hand2')
        save_btn.pack(fill=tk.X, pady=5)
        
        # Status
        self.status_var = tk.StringVar(value="Initializing...")
        ttk.Label(left_frame, textvariable=self.status_var, style='Status.TLabel').pack(pady=5)
        
        # === RIGHT: Camera View ===
        right_frame = ttk.Frame(main)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6), facecolor=bg)
        self.ax.set_facecolor(bg)
        self.ax.set_title("Camera View", color=fg)
        self.ax.tick_params(colors=fg)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Placeholder image
        self.current_image = None
        self.im_display = None
    
    def browse_folder(self):
        folder = filedialog.askdirectory(initialdir=self.bmp_folder)
        if folder:
            self.bmp_folder = Path(folder)
            self.folder_var.set(str(self.bmp_folder))
            self.refresh_bmp_list()
    
    def refresh_bmp_list(self):
        self.bmp_listbox.delete(0, tk.END)
        self.bmp_files = []
        
        folder = Path(self.folder_var.get())
        if folder.exists():
            # Find all BMP files
            files = sorted(folder.glob("*.bmp"))
            for f in files:
                self.bmp_files.append(f)
                # Extract defocus value from filename for display
                name = f.stem
                if "defocus" in name.lower():
                    # Try to extract the defocus value
                    parts = name.split("defocus_")
                    if len(parts) > 1:
                        display = f"Defocus: {parts[1]}"
                    else:
                        display = name
                else:
                    display = name
                self.bmp_listbox.insert(tk.END, display)
        
        self.status_var.set(f"Found {len(self.bmp_files)} BMP files")
    
    def on_bmp_select(self, event):
        selection = self.bmp_listbox.curselection()
        if selection:
            idx = selection[0]
            self.current_bmp = self.bmp_files[idx]
            self.status_var.set(f"Selected: {self.current_bmp.name}")
    
    def init_slm(self):
        if not SLM_AVAILABLE:
            self.status_var.set("SLM SDK not available")
            return
        
        try:
            err = HEDS.SDK.Init(4, 0)
            if err != HEDSERR_NoError:
                raise Exception(HEDS.SDK.ErrorString(err))
            
            self.slm = HEDS.SLM.Init()
            if self.slm.errorCode() != HEDSERR_NoError:
                raise Exception(HEDS.SDK.ErrorString(self.slm.errorCode()))
            
            self.slm_initialized = True
            self.status_var.set("SLM initialized")
        except Exception as e:
            self.status_var.set(f"SLM error: {str(e)[:40]}")
    
    def init_camera(self):
        if not CAMERA_AVAILABLE:
            self.status_var.set("Camera not available")
            return
        
        try:
            self.camera = BaslerCamera(exposure_time_us=self.exposure_us)
            self.camera_initialized = True
            self.status_var.set("Camera initialized")
        except Exception as e:
            self.status_var.set(f"Camera error: {str(e)[:40]}")
    
    def update_exposure(self):
        try:
            new_exp = int(self.exp_var.get())
            if self.camera_initialized and self.camera:
                self.camera.set_exposure(new_exp)
            self.exposure_us = new_exp
            self.status_var.set(f"Exposure set to {new_exp} us")
        except ValueError:
            self.status_var.set("Invalid exposure value")
    
    def load_on_slm(self):
        if self.current_bmp is None:
            self.status_var.set("No BMP selected")
            return
        
        if not self.slm_initialized:
            self.status_var.set("SLM not initialized")
            return
        
        try:
            err, handle = self.slm.loadPhaseDataFromFile(str(self.current_bmp))
            if err != HEDSERR_NoError:
                raise Exception(HEDS.SDK.ErrorString(err))
            
            err = handle.show()
            if err != HEDSERR_NoError:
                raise Exception(HEDS.SDK.ErrorString(err))
            
            self.status_var.set(f"Loaded: {self.current_bmp.name}")
            
            # Auto-capture if enabled
            if self.auto_capture_var.get():
                self.root.after(100, self.capture_image)  # Small delay for SLM to update
                
        except Exception as e:
            self.status_var.set(f"Load error: {str(e)[:40]}")
    
    def capture_image(self):
        if not self.camera_initialized or self.camera is None:
            self.status_var.set("Camera not available")
            # Show placeholder
            self.ax.clear()
            self.ax.text(0.5, 0.5, "Camera not available", 
                        ha='center', va='center', fontsize=14, color='red')
            self.canvas.draw()
            return
        
        try:
            # Update exposure if changed
            try:
                new_exp = int(self.exp_var.get())
                if new_exp != self.exposure_us:
                    self.camera.set_exposure(new_exp)
                    self.exposure_us = new_exp
            except ValueError:
                pass
            
            # Get number of averages
            try:
                self.num_averages = int(self.avg_var.get())
            except ValueError:
                self.num_averages = 1
            
            # Capture
            self.current_image = self.camera.capture_image(num_average=self.num_averages)
            
            # Display
            self.ax.clear()
            self.im_display = self.ax.imshow(self.current_image, cmap='hot', 
                                             interpolation='nearest')
            
            # Add colorbar if not exists
            if not hasattr(self, 'cbar') or self.cbar is None:
                self.cbar = self.fig.colorbar(self.im_display, ax=self.ax)
            else:
                self.cbar.update_normal(self.im_display)
            
            # Title with stats
            max_val = np.max(self.current_image)
            mean_val = np.mean(self.current_image)
            title = f"Max: {max_val:.0f} | Mean: {mean_val:.0f}"
            if self.current_bmp:
                # Extract defocus from filename
                name = self.current_bmp.stem
                if "defocus" in name.lower():
                    parts = name.split("defocus_")
                    if len(parts) > 1:
                        title = f"{parts[1]} | {title}"
            
            self.ax.set_title(title, color='#eef1ff', fontsize=11)
            
            self.canvas.draw()
            self.status_var.set(f"Captured: max={max_val:.0f}, mean={mean_val:.0f}")
            
        except Exception as e:
            self.status_var.set(f"Capture error: {str(e)[:40]}")
    
    def save_image(self):
        if self.current_image is None:
            self.status_var.set("No image to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output folder
        out_dir = self.bmp_folder / "camera_captures"
        out_dir.mkdir(exist_ok=True)
        
        # Filename
        if self.current_bmp:
            base_name = self.current_bmp.stem
        else:
            base_name = "capture"
        
        out_path = out_dir / f"{base_name}_{timestamp}.png"
        
        # Save
        plt.imsave(str(out_path), self.current_image, cmap='hot')
        
        # Also save numpy array
        npy_path = out_dir / f"{base_name}_{timestamp}.npy"
        np.save(str(npy_path), self.current_image)
        
        self.status_var.set(f"Saved: {out_path.name}")
    
    def on_closing(self):
        # Cleanup
        if self.camera:
            try:
                self.camera.close()
            except:
                pass
        
        if self.slm_initialized:
            try:
                HEDS.SDK.Close()
            except:
                pass
        
        self.root.destroy()


def main():
    root = tk.Tk()
    root.geometry("1200x700")
    
    # Check command line for folder
    bmp_folder = None
    if len(sys.argv) > 1:
        bmp_folder = sys.argv[1]
    
    app = DefocusCameraGUI(root, bmp_folder)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
