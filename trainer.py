# trainer.py

import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
import eventlet
import json

# Import the unified model classes from models.py
from models import UNet, UNetWithFiLM, UNetWithPatchFeatures
# Import the single source of truth for feature calculations
from feature_calculators import CMCalc, FMCalc

class Trainer:

    def __init__(self, config, socketio):
        self.config = config
        self.socketio = socketio
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Trainer initialized on device: {self.device}")

    def update_status(self, log_message, stage=None, progress=None):
        print(log_message)
        payload = {'log_message': log_message}
        if stage:
            payload['stage'] = stage
        if progress is not None:
            payload['progress'] = progress
        self.socketio.emit('status_update', payload)
        eventlet.sleep(0.01)

    # DATA PREPROCESSING FUNCTIONS
    def preprocess_images(self, original_image_paths, processed_image_dir, img_size):
        os.makedirs(processed_image_dir, exist_ok=True)
        self.update_status("--- Starting Image Pre-processing ---", stage='setup_images', progress=0)
        
        processed_image_paths = []
        total = len(original_image_paths)
        
        for i, img_path in enumerate(original_image_paths):
            filename = os.path.basename(img_path)
            processed_path = os.path.join(processed_image_dir, os.path.splitext(filename)[0] + '.png')
            processed_image_paths.append(processed_path)
            
            if not os.path.exists(processed_path):
                image_pil = Image.open(img_path).convert("RGB").resize(img_size)
                image_pil.save(processed_path)
            
            if (i + 1) % 5 == 0 or (i + 1) == total:
                progress = int(((i + 1) / total) * 100)
                self.update_status(f"Processed image {i+1}/{total}", stage='setup_images', progress=progress)
                
        self.update_status(" Image pre-processing complete.", stage='setup_images', progress=100)
        return processed_image_paths

    def preprocess_masks(self, rgb_mask_paths, processed_mask_dir, mapping, img_size):
        os.makedirs(processed_mask_dir, exist_ok=True)
        self.update_status("--- Starting Mask Pre-processing ---", stage='setup_masks', progress=0)
        
        processed_mask_paths = []
        total = len(rgb_mask_paths)
        
        for i, rgb_path in enumerate(rgb_mask_paths):
            filename = os.path.basename(rgb_path)
            processed_path = os.path.join(processed_mask_dir, os.path.splitext(filename)[0] + '.png')
            processed_mask_paths.append(processed_path)
            
            if not os.path.exists(processed_path):
                mask_rgb_pil = Image.open(rgb_path).convert("RGB")
                mask_rgb_pil = mask_rgb_pil.resize(img_size, resample=Image.NEAREST)
                mask_rgb_np = np.array(mask_rgb_pil)
                mask_class = np.zeros(mask_rgb_np.shape[:2], dtype=np.uint8)
                
                for class_idx_inner, (rgb_color, class_idx_val) in enumerate(mapping.items()):
                    matches = np.all(mask_rgb_np == np.array(rgb_color), axis=-1)
                    mask_class[matches] = class_idx_val
                    if class_idx_inner % 5 == 0:
                        eventlet.sleep(0) 
                
                cv2.imwrite(processed_path, mask_class)
            
            if (i + 1) % 5 == 0 or (i + 1) == total:
                progress = int(((i + 1) / total) * 100)
                self.update_status(f"Processed mask {i+1}/{total}", stage='setup_masks', progress=progress)
                
        self.update_status(" Mask pre-processing complete.", stage='setup_masks', progress=100)
        return processed_mask_paths

    def create_color_maps(self, csv_path):
        df = pd.read_csv(csv_path)
        df.columns = [col.strip() for col in df.columns]
        rgb_to_class_idx = {tuple([row['r'], row['g'], row['b']]): i for i, row in df.iterrows()}
        num_classes = len(df)
        return rgb_to_class_idx, num_classes

    # --- [REMOVED] The incorrect InvariantMethods class is gone. ---

    class SegmentationDataset(Dataset):
        def __init__(self, image_paths, mask_paths, feature_cache=None, feature_type='none'):
            self.image_paths = image_paths
            self.mask_paths = mask_paths
            self.feature_cache = feature_cache
            self.feature_type = feature_type

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            mask_path = self.mask_paths[idx]
            image = Image.open(img_path).convert("RGB")
            image_tensor = TF.to_tensor(image)
            mask = Image.open(mask_path)
            mask_np = np.array(mask)
            mask_tensor = torch.from_numpy(mask_np).long()
            
            if self.feature_type == 'none':
                return image_tensor, mask_tensor
            else:
                feature_data = self.feature_cache[img_path]
                feature_tensor = torch.from_numpy(feature_data).float()
                if self.feature_type == 'patch':
                    feature_tensor = feature_tensor.permute(2, 0, 1)
                return (image_tensor, feature_tensor), mask_tensor

    # TRAINING & VALIDATION LOOPS
    def train_fn(self, loader, model, optimizer, loss_fn, feature_type='none', epoch_num=0, total_epochs=0):
        model.train()
        total_loss = 0
        num_batches = len(loader)
        
        for i, (data, targets) in enumerate(loader):
            targets = targets.to(device=self.device)
            
            if feature_type != 'none':
                images, features = data
                images, features = images.to(self.device), features.to(self.device)
                predictions = model(images, features)
            else:
                images = data.to(self.device)
                predictions = model(images)
                
            loss = loss_fn(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            progress = int(((i + 1) / num_batches) * 100)
            log_msg = f"Epoch {epoch_num+1}/{total_epochs} - Batch {i+1}/{num_batches} - Loss: {loss.item():.4f}"
            self.socketio.emit('batch_update', {'progress': progress, 'log': log_msg})
            eventlet.sleep(0) 
            
        return total_loss / len(loader)

    def check_accuracy(self, loader, model, feature_type='none'):
        num_correct, num_pixels = 0, 0
        model.eval()
        with torch.no_grad():
            for data, targets in loader:
                targets_on_device = targets.to(self.device)
                
                if feature_type != 'none':
                    images, features = data
                    images, features = images.to(self.device), features.to(self.device)
                    preds = model(images, features)
                else:
                    images = data.to(self.device)
                    preds = model(images)
                    
                preds = torch.argmax(preds, dim=1)
                num_correct += (preds == targets_on_device).sum()
                num_pixels += torch.numel(preds)
                
        accuracy = (num_correct / num_pixels) * 100
        self.update_status(f"Validation accuracy: {accuracy:.2f}%")
        return accuracy.item()
    
    # MAIN EXECUTION METHOD
    def run(self):
        try:
            self.update_status("Starting training process...", stage='init', progress=0)
            
            # Setup Directories
            output_dir = self.config['output_dir']
            saved_models_dir = os.path.join(output_dir, "saved_models")
            processed_image_dir = os.path.join(output_dir, "processed_images")
            processed_mask_dir = os.path.join(output_dir, "processed_masks")
            cache_dir = os.path.join(output_dir, "cache")
            os.makedirs(saved_models_dir, exist_ok=True)
            os.makedirs(processed_image_dir, exist_ok=True)
            os.makedirs(processed_mask_dir, exist_ok=True)
            os.makedirs(cache_dir, exist_ok=True)
            self.update_status(" Directories created.")
            
            # Load Data and Mappings
            rgb_to_class_idx, num_classes = self.create_color_maps(self.config['class_csv'])
            self.update_status(f" Loaded color mappings for {num_classes} classes.")
            all_image_files = sorted(glob(os.path.join(self.config['image_dir'], '*.jpg')))
            all_rgb_mask_files = sorted(glob(os.path.join(self.config['mask_dir'], '*.png')))
            subset_size = self.config['data_subset'] if self.config['data_subset'] > 0 else len(all_image_files)
            image_files_original = all_image_files[:subset_size]
            rgb_mask_files = all_rgb_mask_files[:subset_size]
            
            # Pre-process Data
            target_size = (self.config['img_size'], self.config['img_size'])
            processed_image_files = self.preprocess_images(image_files_original, processed_image_dir, target_size)
            processed_mask_files = self.preprocess_masks(rgb_mask_files, processed_mask_dir, rgb_to_class_idx, target_size)
            
            # Pre-compute/cache Invariant Features using the centralized calculators
            cm_calc = CMCalc()
            fm_calc = FMCalc()

            def compute_patches(image_rgb, method_func, patch_size=64, stride=16):
                h, w, _ = image_rgb.shape
                output_h = (h - patch_size) // stride + 1
                output_w = (w - patch_size) // stride + 1
                dummy_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                feature_len = len(method_func(dummy_patch))
                feature_map = np.zeros((output_h, output_w, feature_len))
                for i in range(output_h):
                    for j in range(output_w):
                        patch = image_rgb[i * stride: i * stride + patch_size, j * stride: j * stride + patch_size]
                        feature_map[i, j, :] = method_func(patch)
                return feature_map
            
            feature_caches = {}
            n_fm_moments_per_channel = 4096 # Use a fixed, reasonable number like 64*64

            feature_configs = {
                'cm_global': (cm_calc.cm_rgb, 'global'),
                'fmt_global': (lambda img: fm_calc.fm_rgb(img, n=n_fm_moments_per_channel), 'global'),
                'cm_patch': (lambda img: compute_patches(img, cm_calc.cm_rgb), 'patch'),
            }
            for name, (func, f_type) in feature_configs.items():
                cache_path = os.path.join(cache_dir, f"{name}_cache_{subset_size}.pkl")
                if os.path.exists(cache_path):
                    self.update_status(f" Loading {name} features from cache...")
                    with open(cache_path, 'rb') as f:
                        feature_caches[name] = pickle.load(f)
                else:
                    self.update_status(f" {name} cache not found. Pre-computing...", stage='setup_caching', progress=0)
                    cache_data = {}
                    total_files = len(processed_image_files)
                    for i, img_path in enumerate(tqdm(processed_image_files, desc=f"Computing {name}")):
                        cache_data[img_path] = func(np.array(Image.open(img_path)))
                        if (i + 1) % 5 == 0 or (i + 1) == total_files:
                            progress = int(((i + 1) / total_files) * 100)
                            self.update_status(f"Caching feature {i+1}/{total_files}", stage='setup_caching', progress=progress)
                    with open(cache_path, 'wb') as f:
                        pickle.dump(cache_data, f)
                    feature_caches[name] = cache_data
                    self.update_status(f" {name} features saved to cache.", stage='setup_caching', progress=100)
            
            # Split Data and Create Loaders 
            X_train, X_val, y_train, y_val = train_test_split(processed_image_files, processed_mask_files, test_size=0.2, random_state=42)
            self.update_status(f" Data split: {len(X_train)} training, {len(X_val)} validation.")
            loader_args = {'batch_size': self.config['batch_size']}
            if self.device.type == "cuda":
                loader_args.update({'num_workers': 2, 'pin_memory': True})
                
            # Define Models to Train 
            cm_global_len = next(iter(feature_caches['cm_global'].values())).shape[0]
            fmt_global_len = next(iter(feature_caches['fmt_global'].values())).shape[0]
            cm_patch_len = next(iter(feature_caches['cm_patch'].values())).shape[-1]

            models_to_train = {
                "Baseline": {"model": UNet(num_classes=num_classes, features=self.config['unet_features']), "cache": None, "type": "none"},
                "FiLM_CM": {"model": UNetWithFiLM(num_classes=num_classes, feature_len=cm_global_len, features=self.config['unet_features'], film_hidden_dim=self.config['film_hidden_dim']), "cache": feature_caches['cm_global'], "type": "global"},
                "FiLM_FMT": {"model": UNetWithFiLM(num_classes=num_classes, feature_len=fmt_global_len, features=self.config['unet_features'], film_hidden_dim=self.config['film_hidden_dim']), "cache": feature_caches['fmt_global'], "type": "global"},
                "Patch_CM": {"model": UNetWithPatchFeatures(num_classes=num_classes, feature_len=cm_patch_len, features=self.config['unet_features'], patch_hidden_dim=self.config['patch_hidden_dim']), "cache": feature_caches['cm_patch'], "type": "patch"},
            }
            
            # Main Training Loop
            total_epochs = self.config['epochs']
            histories = {}
            for name, model_config in models_to_train.items():
                self.update_status("\n" + "="*15 + f" Training {name} " + "="*15, stage='training')
                
                train_dset = self.SegmentationDataset(X_train, y_train, model_config['cache'], model_config['type'])
                val_dset = self.SegmentationDataset(X_val, y_val, model_config['cache'], model_config['type'])
                train_loader = DataLoader(train_dset, shuffle=True, **loader_args)
                val_loader = DataLoader(val_dset, shuffle=False, **loader_args)
                
                model = model_config['model'].to(self.device)
                optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
                loss_fn = nn.CrossEntropyLoss()
                
                history = {'loss': [], 'val_accuracy': []}
                for epoch in range(total_epochs):
                    self.socketio.emit('epoch_update', {'model': name, 'epoch': epoch+1, 'total_epochs': total_epochs})
                    loss = self.train_fn(train_loader, model, optimizer, loss_fn, model_config['type'], epoch, total_epochs)
                    acc = self.check_accuracy(val_loader, model, model_config['type'])
                    history['loss'].append(loss)
                    history['val_accuracy'].append(acc)
                    self.socketio.emit('chart_update', {'model': name, 'epoch': epoch + 1, 'loss': loss, 'accuracy': acc})
                    
                histories[name] = history
                torch.save(model.state_dict(), os.path.join(saved_models_dir, f"{name}_model.pth"))
                self.update_status(f" {name} model saved.")

            # Save the configuration used for this training run
            self.update_status("--- Saving training configuration ---")
            config_to_save = {k: v for k, v in self.config.items()}
            config_path = os.path.join(output_dir, "training_config.json")
            with open(config_path, 'w') as f:
                json.dump(config_to_save, f, indent=4)
            self.update_status(f"✅ Configuration saved to {config_path}")

            # Final Visualization 
            self.update_status("--- Generating final training plots ---")
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            for name, history in histories.items():
                ax1.plot(history['val_accuracy'], label=f'{name}', marker='o')
            ax1.set_title(f'Validation Accuracy (N={subset_size})')
            ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy (%)'); ax1.legend(); ax1.grid(True)
            for name, history in histories.items():
                ax2.plot(history['loss'], label=f'{name}', marker='.')
            ax2.set_title(f'Training Loss (N={subset_size})')
            ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.legend(); ax2.grid(True)
            plt.tight_layout()
            
            plot_path_relative = os.path.join('outputs', f"training_results_{subset_size}.png")
            plot_path_full = os.path.join('static', plot_path_relative)
            plt.savefig(plot_path_full)
            self.update_status(f" Final plot saved to {plot_path_full}")
            self.socketio.emit('training_complete', {'plot_url': plot_path_relative})
            
        except Exception as e:
            self.update_status(f" An error occurred: {e}")
            self.socketio.emit('training_error', {'error': str(e)})