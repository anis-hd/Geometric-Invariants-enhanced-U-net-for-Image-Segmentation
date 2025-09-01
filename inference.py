# inference.py

import os
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import pandas as pd
import pickle
import eventlet
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time

from models import UNet, UNetWithFiLM, UNetWithPatchFeatures
# Import the single source of truth for feature calculations
from feature_calculators import InvariantMethods

class InferenceRunner:
    """Handles the model inference process for a single image."""

    def __init__(self, config, socketio):
        self.config = config
        self.socketio = socketio
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.update_log("Inference Runner initialized.")

    def update_log(self, message):
        print(message)
        self.socketio.emit('inference_log', {'data': message})
        eventlet.sleep(0.01)

    # --- Helper methods ---
    def _create_color_maps(self, csv_path):
        df = pd.read_csv(csv_path)
        df.columns = [col.strip() for col in df.columns]
        class_idx_to_rgb = {i: np.array([row['r'], row['g'], row['b']]) for i, row in df.iterrows()}
        return class_idx_to_rgb, len(df)

    def _mask_to_rgb(self, mask_np, cmap):
        rgb_mask = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
        for class_idx, color in cmap.items():
            rgb_mask[mask_np == class_idx] = color
        return Image.fromarray(rgb_mask)

    def _plot_invariants(self, moments, title, method='cm'):
        fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
        n = len(moments)
        x = np.arange(n)
        ax.bar(x, moments, color='steelblue', width=1.0)
        ax.set_title(title)
        ax.set_xlabel("Feature Index")
        if method == 'fm':
            ax.set_ylabel("Magnitude")
            # For the jupyter version, a simple y-scale is fine
        else:
            ax.set_ylabel("Magnitude (log scale)")
            ax.set_yscale('symlog')
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

    def run(self):
        try:
            # 1. Load the saved training config first
            self.update_log("--- 1. Loading Training Configuration ---")
            config_path = os.path.join(self.config['output_dir'], "training_config.json")
            if not os.path.exists(config_path):
                error_msg = f"Configuration file not found at {config_path}."
                self.update_log(f"🔴 ERROR: {error_msg}")
                self.socketio.emit('inference_error', {'error': error_msg})
                return
            with open(config_path, 'r') as f:
                training_config = json.load(f)
            keys_to_override = ['unet_features', 'film_hidden_dim', 'patch_hidden_dim', 'img_size']
            for key in keys_to_override:
                if key in training_config:
                    self.config[key] = training_config[key]
            self.update_log("✅ Successfully loaded and applied training configuration.")
            
            # 2. Setup
            self.update_log("--- 2. Initializing ---")
            img_path = self.config['image_path']
            output_dir = self.config['output_dir']
            class_csv = self.config['class_csv']
            img_size = (self.config['img_size'], self.config['img_size'])
            saved_models_dir = os.path.join(output_dir, "saved_models")
            
            color_map, num_classes = self._create_color_maps(class_csv)
            self.update_log(f"Loaded color map with {num_classes} classes.")

            # 3. Load and Preprocess Image
            self.update_log(f"--- 3. Loading and Processing Image: {os.path.basename(img_path)} ---")
            image = Image.open(img_path).convert("RGB").resize(img_size)
            image_np = np.array(image)
            image_tensor = TF.to_tensor(image).unsqueeze(0).to(self.device)

            # 4. Calculate Invariants using the Jupyter-identical methods
            self.update_log("--- 4. Calculating Geometric Invariants (Jupyter Method) ---")
            inv_methods = InvariantMethods()
            
            cm_global = inv_methods.compute_complex_moments_rgb(image_np)
            fmt_global = inv_methods.fourier_mellin_descriptor(image_np)
            # Correctly calculate patch features for the input image
            cm_patch = inv_methods.compute_invariants_on_patches(image_np, inv_methods.compute_complex_moments_rgb)

            self.update_log(" Invariants calculated successfully.")
            
            features = {
                'cm_global': torch.from_numpy(cm_global).float().unsqueeze(0).to(self.device),
                'fmt_global': torch.from_numpy(fmt_global).float().unsqueeze(0).to(self.device),
                'cm_patch': torch.from_numpy(cm_patch).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            }

            # 5. Load Models
            self.update_log("--- 5. Loading Pre-trained Models ---")
            models_to_run = {
                "Baseline": {"model": UNet(num_classes=num_classes, features=self.config['unet_features']), "features": None},
                "FiLM_CM": {"model": UNetWithFiLM(num_classes=num_classes, feature_len=cm_global.shape[0], features=self.config['unet_features'], film_hidden_dim=self.config['film_hidden_dim']), "features": features['cm_global']},
                "FiLM_FMT": {"model": UNetWithFiLM(num_classes=num_classes, feature_len=fmt_global.shape[0], features=self.config['unet_features'], film_hidden_dim=self.config['film_hidden_dim']), "features": features['fmt_global']},
                "Patch_CM": {"model": UNetWithPatchFeatures(num_classes=num_classes, feature_len=cm_patch.shape[-1], features=self.config['unet_features'], patch_hidden_dim=self.config['patch_hidden_dim']), "features": features['cm_patch']},
            }
            
            for name, config in models_to_run.items():
                model_path = os.path.join(saved_models_dir, f"{name}_model.pth")
                if not os.path.exists(model_path):
                    self.update_log(f"⚠️ WARNING: Model file not found for '{name}' at {model_path}. Skipping.")
                    config['model'] = None
                    continue
                config['model'].load_state_dict(torch.load(model_path, map_location=self.device))
                config['model'].to(self.device)
                config['model'].eval()
            self.update_log("Models loaded.")

            # 6. Run Inference and Generate Outputs
            self.update_log("--- 6. Running Inference on Models ---")
            results = {'segmentations': []}
            with torch.no_grad():
                for name, config in models_to_run.items():
                    if config['model'] is None:
                        continue
                    
                    self.update_log(f"  - Predicting with {name}...")
                    if config['features'] is not None:
                        preds = config['model'](image_tensor, config['features'])
                    else:
                        preds = config['model'](image_tensor)
                    
                    pred_mask_np = torch.argmax(preds, dim=1).squeeze().cpu().numpy()
                    pred_mask_rgb = self._mask_to_rgb(pred_mask_np, color_map)
                    
                    filename = f"inference_{name}_{time.time()}.png"
                    save_path = os.path.join('static', 'outputs', filename)
                    pred_mask_rgb.save(save_path)
                    
                    results['segmentations'].append({
                        'name': name,
                        'path': os.path.join('outputs', filename).replace('\\', '/')
                    })
            
            # 7. Compile and Send Final Results
            self.update_log("--- 7. Compiling Final Results ---")
            results['invariants'] = [
                {'name': 'Complex Moments (Global)', 'plot_b64': self._plot_invariants(cm_global, 'Complex Moments', method='cm')},
                {'name': 'Fourier-Mellin (Global)', 'plot_b64': self._plot_invariants(fmt_global, 'Fourier-Mellin', method='fm')}
            ]
            self.socketio.emit('inference_result', results)
            self.update_log("✅ Inference complete.")
            self.socketio.emit('inference_complete')

        except Exception as e:
            error_message = f"An error occurred during inference: {e}"
            self.update_log(f"🔴 ERROR: {error_message}")
            self.socketio.emit('inference_error', {'error': str(e)})