#
#  TRAINING LOGIC MODULE 
#
# This module contains the Trainer class, which encapsulates all logic
# for data preprocessing, model training, and real-time communication
# with the Flask web interface.
# =====================================================================================




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
        """
        Resizes and saves source images to a processed directory. Skips existing files.
        """
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
            
            # Periodically update the frontend
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
                
                # Resize mask to match image size, using NEAREST neighbor interpolation
                mask_rgb_pil = mask_rgb_pil.resize(img_size, resample=Image.NEAREST)

                mask_rgb_np = np.array(mask_rgb_pil)
                mask_class = np.zeros(mask_rgb_np.shape[:2], dtype=np.uint8)
                
                # This inner loop is computationally expensive, so we must yield inside it
                for class_idx_inner, (rgb_color, class_idx_val) in enumerate(mapping.items()):
                    matches = np.all(mask_rgb_np == np.array(rgb_color), axis=-1)
                    mask_class[matches] = class_idx_val
                    if class_idx_inner % 5 == 0:
                        eventlet.sleep(0) 
                
                cv2.imwrite(processed_path, mask_class)
            
            if (i + 1) % 5 == 0 or (i + 1) == total:
                progress = int(((i + 1) / total) * 100)
                self.update_status(f"Processed mask {i+1}/{total}", stage='setup_masks', progress=progress)
                
        self.update_status("✅ Mask pre-processing complete.", stage='setup_masks', progress=100)
        return processed_mask_paths
     # Mapping from RGB to class indices
    def create_color_maps(self, csv_path):

        df = pd.read_csv(csv_path)
        df.columns = [col.strip() for col in df.columns]
        rgb_to_class_idx = {tuple([row['r'], row['g'], row['b']]): i for i, row in df.iterrows()}
        num_classes = len(df)
        return rgb_to_class_idx, num_classes

    # HELPER & MODEL CLASSES


    class InvariantMethods:
        """Container for geometric invariant calculation methods."""
        def _compute_central_moments(self, channel):
            h, w = channel.shape
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            m00 = np.sum(channel)
            if m00 == 0: return 0, 0, m00
            x_bar = np.sum(x_coords * channel) / m00
            y_bar = np.sum(y_coords * channel) / m00
            return x_bar, y_bar, m00
        


        # Complex moments
        def compute_complex_moments_rgb(self, image_rgb, max_order: int = 2):
            all_channel_moments = []
            for c in range(image_rgb.shape[2]):
                channel = image_rgb[:, :, c].astype(np.float64)
                h, w = channel.shape
                x_bar, y_bar, m00 = self._compute_central_moments(channel)
                y_coords, x_coords = np.mgrid[0:h, 0:w]
                z = (x_coords - x_bar) + 1j * (y_coords - y_bar)
                moments = []
                m00_norm = m00 if m00 != 0 else 1
                for p in range(max_order + 1):
                    for q in range(max_order + 1):
                        if p + q >= 2:
                            cm_raw = np.sum(channel * (z ** p) * (np.conj(z) ** q))
                            norm_factor = m00_norm ** (((p + q) / 2.0) + 1.0)
                            moments.append(cm_raw / norm_factor if norm_factor != 0 else 0)
                all_channel_moments.append(np.abs(np.array(moments)))
            return np.concatenate(all_channel_moments)
        




        # Fourier-Mellin
        def fourier_mellin_descriptor(self, image_rgb, log_polar_size=(64, 64)):
            descriptors = []
            for c in range(image_rgb.shape[2]):
                channel = image_rgb[:, :, c].astype(np.float32)
                f_transform = np.fft.fft2(channel)
                f_transform_shifted = np.fft.fftshift(f_transform)
                magnitude_spectrum = np.log1p(np.abs(f_transform_shifted))
                center = (channel.shape[1] / 2, channel.shape[0] / 2)
                max_radius = np.sqrt(((channel.shape[0] / 2) ** 2) + ((channel.shape[1] / 2) ** 2))
                log_polar_map = cv2.logPolar(magnitude_spectrum, center, max_radius, flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
                log_polar_resized = cv2.resize(log_polar_map, log_polar_size, interpolation=cv2.INTER_LINEAR)
                descriptors.append(np.abs(np.fft.fft2(log_polar_resized)).flatten())
            return np.concatenate(descriptors)

        def compute_invariants_on_patches(self, image_rgb, method_func, patch_size=64, stride=16):
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

    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        def forward(self, x):
            return self.conv(x)
        





############# Baseline U-net architecture





    class UNet(nn.Module):
        def __init__(self, in_channels=3, num_classes=23, features=[16, 32, 64, 128]):
            super().__init__()
            self.downs = nn.ModuleList()
            self.ups = nn.ModuleList()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
            for feature in features:
                self.downs.append(Trainer.DoubleConv(in_channels, feature))
                in_channels = feature
                
            for feature in reversed(features):
                self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
                self.ups.append(Trainer.DoubleConv(feature * 2, feature))
                
            self.bottleneck = Trainer.DoubleConv(features[-1], features[-1] * 2)
            self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

        def forward(self, x):
            skip_connections = []
            for down in self.downs:
                x = down(x)
                skip_connections.append(x)
                x = self.pool(x)
            
            x = self.bottleneck(x)
            skip_connections = skip_connections[::-1]
            
            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)
                skip_connection = skip_connections[idx // 2]
                if x.shape != skip_connection.shape:
                    x = TF.resize(x, size=skip_connection.shape[2:])
                concat_skip = torch.cat((skip_connection, x), dim=1)
                x = self.ups[idx + 1](concat_skip)
            
            return self.final_conv(x)



# U-net with feature wise linear modulation (FILM)
    class UNetWithFiLM(UNet):
        def __init__(self, in_channels=3, num_classes=23, features=[16, 32, 64, 128], feature_len=30):
            super().__init__(in_channels=in_channels, num_classes=num_classes, features=features)
            bottleneck_channels = features[-1]
            self.feature_processor = nn.Sequential(
                nn.Linear(feature_len, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, bottleneck_channels * 2) # Predict gamma and beta
            )
            self.bottleneck = Trainer.DoubleConv(bottleneck_channels, features[-1] * 2)

        def forward(self, x, ft):
            skip_connections = []
            for down in self.downs:
                x = down(x)
                skip_connections.append(x)
                x = self.pool(x)
            
            #FiLM modulation
            film_params = self.feature_processor(ft)
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
            gamma = gamma.view(gamma.size(0), -1, 1, 1)
            beta = beta.view(beta.size(0), -1, 1, 1)
            x = gamma * x + beta
            
            x = self.bottleneck(x)
            skip_connections = skip_connections[::-1]
            
            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)
                skip_connection = skip_connections[idx // 2]
                if x.shape != skip_connection.shape:
                    x = TF.resize(x, size=skip_connection.shape[2:])
                concat_skip = torch.cat((skip_connection, x), dim=1)
                x = self.ups[idx + 1](concat_skip)
            
            return self.final_conv(x)



#Unet with patch concatenation
    class UNetWithPatchFeatures(UNet):
        def __init__(self, in_channels=3, num_classes=23, features=[16, 32, 64, 128], feature_len=30):
            super().__init__(in_channels=in_channels + features[0], num_classes=num_classes, features=features)
            self.patch_processor = nn.Sequential(
                nn.Conv2d(feature_len, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, features[0], kernel_size=1)
            )

        def forward(self, x, ft_map):
            processed_ft = self.patch_processor(ft_map)
            processed_ft = TF.resize(processed_ft, size=x.shape[2:])
            x = torch.cat([x, processed_ft], dim=1)
            return super().forward(x)






    #  TRAINING & VALIDATION LOOPS

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
            eventlet.sleep(0) # Critical yield after each batch
            
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
        """The main entry point to start the entire training pipeline."""
        try:
            self.update_status("Starting training process...", stage='init', progress=0)
            
            # --- 1. Setup Directories ---
            output_dir = self.config['output_dir']
            saved_models_dir = os.path.join(output_dir, "saved_models")
            processed_image_dir = os.path.join(output_dir, "processed_images")
            processed_mask_dir = os.path.join(output_dir, "processed_masks")
            cache_dir = os.path.join(output_dir, "cache")
            os.makedirs(saved_models_dir, exist_ok=True)
            os.makedirs(processed_image_dir, exist_ok=True)
            os.makedirs(processed_mask_dir, exist_ok=True)
            os.makedirs(cache_dir, exist_ok=True)
            self.update_status("✅ Directories created.")
            
            # Load Data and Mappings
            rgb_to_class_idx, num_classes = self.create_color_maps(self.config['class_csv'])
            self.update_status(f"✅ Loaded color mappings for {num_classes} classes.")
            all_image_files = sorted(glob(os.path.join(self.config['image_dir'], '*.jpg')))
            all_rgb_mask_files = sorted(glob(os.path.join(self.config['mask_dir'], '*.png')))
            subset_size = self.config['data_subset'] if self.config['data_subset'] > 0 else len(all_image_files)
            image_files_original = all_image_files[:subset_size]
            rgb_mask_files = all_rgb_mask_files[:subset_size]
            
            #Pre-process Data
            target_size = (self.config['img_size'], self.config['img_size'])
            processed_image_files = self.preprocess_images(image_files_original, processed_image_dir, target_size)
            processed_mask_files = self.preprocess_masks(rgb_mask_files, processed_mask_dir, rgb_to_class_idx, target_size)
            
            # Pre-compute/cache Invariant Features
            invariant_extractor = self.InvariantMethods()
            feature_caches = {}
            feature_configs = {
                'cm_global': (invariant_extractor.compute_complex_moments_rgb, 'global'),
                'fmt_global': (invariant_extractor.fourier_mellin_descriptor, 'global'),
                'cm_patch': (lambda img: invariant_extractor.compute_invariants_on_patches(img, invariant_extractor.compute_complex_moments_rgb), 'patch'),
            }
            for name, (func, f_type) in feature_configs.items():
                cache_path = os.path.join(cache_dir, f"{name}_cache_{subset_size}.pkl")
                if os.path.exists(cache_path):
                    self.update_status(f"✅ Loading {name} features from cache...")
                    with open(cache_path, 'rb') as f:
                        feature_caches[name] = pickle.load(f)
                else:
                    self.update_status(f"⚠️ {name} cache not found. Pre-computing...", stage='setup_caching', progress=0)
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
                    self.update_status(f"✅ {name} features saved to cache.", stage='setup_caching', progress=100)
            
            # Split Data and Create Loaders 
            X_train, X_val, y_train, y_val = train_test_split(processed_image_files, processed_mask_files, test_size=0.2, random_state=42)
            self.update_status(f"✅ Data split: {len(X_train)} training, {len(X_val)} validation.")
            loader_args = {'batch_size': self.config['batch_size']}
            if self.device.type == "cuda":
                loader_args.update({'num_workers': 2, 'pin_memory': True})
                
            # Define Models to Train 
            cm_global_len = next(iter(feature_caches['cm_global'].values())).shape[0]
            fmt_global_len = next(iter(feature_caches['fmt_global'].values())).shape[0]
            cm_patch_len = next(iter(feature_caches['cm_patch'].values())).shape[-1]
            models_to_train = {
                "Baseline": {"model": self.UNet(num_classes=num_classes), "cache": None, "type": "none"},
                "FiLM_CM": {"model": self.UNetWithFiLM(num_classes=num_classes, feature_len=cm_global_len), "cache": feature_caches['cm_global'], "type": "global"},
                "FiLM_FMT": {"model": self.UNetWithFiLM(num_classes=num_classes, feature_len=fmt_global_len), "cache": feature_caches['fmt_global'], "type": "global"},
                "Patch_CM": {"model": self.UNetWithPatchFeatures(num_classes=num_classes, feature_len=cm_patch_len), "cache": feature_caches['cm_patch'], "type": "patch"},
            }
            
            #Main Training Loop
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
                self.update_status(f"✅ {name} model saved.")
                
            #Final Visualization 
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
            self.update_status(f"✅ Final plot saved to {plot_path_full}")
            self.socketio.emit('training_complete', {'plot_url': plot_path_relative})
            
        except Exception as e:
            self.update_status(f"🔴 An error occurred: {e}")
            self.socketio.emit('training_error', {'error': str(e)})