import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from glob import glob
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import pickle
import random
import eventlet
import json
from models import UNet, UNetWithFiLM, UNetWithPatchFeatures

class Benchmarker:

    def __init__(self, config, socketio):
        self.config = config
        self.socketio = socketio
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = self.config['output_dir']
        self.update_log("Benchmarker initialized.")

    def update_log(self, message):
        print(message)
        self.socketio.emit('benchmark_log', {'data': message})
        eventlet.sleep(0.01)

    def create_color_maps(self, csv_path):
        df = pd.read_csv(csv_path)
        df.columns = [col.strip() for col in df.columns]
        class_idx_to_rgb = {i: np.array([row['r'], row['g'], row['b']]) for i, row in df.iterrows()}
        num_classes = len(df)
        return class_idx_to_rgb, num_classes

    class EvaluationDataset(Dataset):
        def __init__(self, image_paths, mask_paths, feature_cache=None, feature_type='none', transform=None, img_size=(256, 256)):
            self.image_paths = image_paths
            self.mask_paths = mask_paths
            self.feature_cache = feature_cache
            self.feature_type = feature_type
            self.transform = transform
            self.img_size = img_size

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            mask_path = self.mask_paths[idx]
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            
            if self.transform:
                seed = random.randint(0, 2**32)
                random.seed(seed)
                torch.manual_seed(seed)
                image = self.transform(image)
                random.seed(seed)
                torch.manual_seed(seed)
                mask = self.transform(mask)
            
            image_tensor = TF.to_tensor(image)
            mask_tensor = torch.from_numpy(np.array(mask)).long()
            
            if self.feature_type == 'none':
                return image_tensor, mask_tensor
            else:
                feature_data = self.feature_cache[img_path]
                feature_tensor = torch.from_numpy(feature_data).float()
                if self.feature_type == 'patch':
                    feature_tensor = feature_tensor.permute(2, 0, 1)
                return (image_tensor, feature_tensor), mask_tensor

    def iou_dice_score(self, preds, targets, num_classes, smooth=1e-6):
        iou_scores, dice_scores = [], []
        preds = torch.argmax(preds, dim=1).view(-1)
        targets = targets.view(-1)
        for cls in range(num_classes):
            pred_inds, target_inds = (preds == cls), (targets == cls)
            intersection = (pred_inds & target_inds).sum().float()
            union = (pred_inds | target_inds).sum().float()
            iou = (intersection + smooth) / (union + smooth)
            dice = (2. * intersection + smooth) / (pred_inds.sum() + target_inds.sum() + smooth)
            iou_scores.append(iou.item())
            dice_scores.append(dice.item())
        return np.mean(iou_scores), np.mean(dice_scores)
    
    def evaluate_model(self, loader, model, num_classes, feature_type):
        num_correct, num_pixels = 0, 0
        total_iou, total_dice = 0, 0
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
                
                pred_labels = torch.argmax(preds, dim=1)
                num_correct += (pred_labels == targets_on_device).sum()
                num_pixels += torch.numel(pred_labels)
                iou, dice = self.iou_dice_score(preds, targets_on_device, num_classes)
                total_iou += iou
                total_dice += dice
                
        accuracy = (num_correct / num_pixels) * 100
        mean_iou = (total_iou / len(loader)) * 100
        mean_dice = (total_dice / len(loader)) * 100
        return accuracy.item(), mean_iou, mean_dice
    
    def plot_and_save(self, plot_func, *args, filename):
        plt.figure()
        plot_func(*args)
        relative_path = os.path.join('outputs', filename)
        full_path = os.path.join('static', relative_path)
        plt.savefig(full_path, bbox_inches='tight')
        plt.close()
        self.update_log(f"{filename} generated.")
        self.socketio.emit('benchmark_result', {'type': 'plot', 'url': relative_path, 'title': filename.replace('_', ' ').replace('.png', '').title()})

    def _tsne_plotter(self, features_cache, all_mask_paths, title):
        paths_in_cache = list(features_cache.keys())
        features = np.array([features_cache[p] for p in paths_in_cache])
        def get_dominant_class(mask_path, bg_class=0):
            mask = np.array(Image.open(mask_path))
            unique, counts = np.unique(mask, return_counts=True)
            counts = counts[unique != bg_class]
            unique = unique[unique != bg_class]
            return unique[np.argmax(counts)] if len(counts) > 0 else bg_class
        labels = [get_dominant_class(os.path.join(self.config['processed_mask_dir'], f"{os.path.basename(p).split('.')[0]}.png")) for p in paths_in_cache]
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
        features_2d = tsne.fit_transform(features)
        sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=labels, palette='tab20', legend='full')
        plt.title(title)
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")

    def _comparison_grid_plotter(self, sample_data, models_dict, color_map):
        original_image = Image.open(sample_data['img_path']).convert("RGB")
        original_mask = Image.open(sample_data['mask_path']).convert("L")
        transform = T.RandomAffine(degrees=45, translate=(0.15, 0.15), scale=(0.8, 1.2), shear=10)
        transformed_image = transform(original_image)
        transformed_mask = transform(original_mask)
        img_tensor_orig = TF.to_tensor(original_image).unsqueeze(0).to(self.device)
        img_tensor_trans = TF.to_tensor(transformed_image).unsqueeze(0).to(self.device)
        def mask_to_rgb(mask_np, cmap):
            rgb_mask = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
            for class_idx, color in cmap.items():
                rgb_mask[mask_np == class_idx] = color
            return rgb_mask
        preds = {}
        with torch.no_grad():
            for name, config in models_dict.items():
                model = config['model']
                model.eval()
                feature_type = config['feature_type']
                features_orig = None
                if feature_type != 'none':
                    feature_tensor = torch.from_numpy(sample_data[config['cache_key']]).float().unsqueeze(0).to(self.device)
                    if feature_type == 'patch':
                        feature_tensor = feature_tensor.permute(0, 3, 1, 2)
                    features_orig = feature_tensor
                pred_orig = model(img_tensor_orig, features_orig) if feature_type != 'none' else model(img_tensor_orig)
                pred_trans = model(img_tensor_trans, features_orig) if feature_type != 'none' else model(img_tensor_trans)
                preds[name] = {'orig': mask_to_rgb(torch.argmax(pred_orig, dim=1).squeeze().cpu().numpy(), color_map), 'trans': mask_to_rgb(torch.argmax(pred_trans, dim=1).squeeze().cpu().numpy(), color_map)}
        n_models = len(models_dict)
        fig, axes = plt.subplots(2, n_models + 2, figsize=(24, 8))
        plt.suptitle("Model Segmentation Invariance Comparison", fontsize=20)
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title("Original Image")
        axes[0, 1].imshow(mask_to_rgb(np.array(original_mask), color_map))
        axes[0, 1].set_title("Ground Truth")
        for i, name in enumerate(models_dict.keys()):
            axes[0, i + 2].imshow(preds[name]['orig'])
            axes[0, i + 2].set_title(f"{name}\n(Original)")
        axes[1, 0].imshow(transformed_image)
        axes[1, 0].set_title("Transformed Image")
        axes[1, 1].imshow(mask_to_rgb(np.array(transformed_mask), color_map))
        axes[1, 1].set_title("Transformed GT")
        for i, name in enumerate(models_dict.keys()):
            axes[1, i + 2].imshow(preds[name]['trans'])
            axes[1, i + 2].set_title(f"{name}\n(Transformed)")
        for ax in axes.flat:
            ax.axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def _robustness_plotter(self, results_dict, metric):
        model_names = list(results_dict.keys())
        transform_names = list(next(iter(results_dict.values())).keys())
        n_groups = len(transform_names)
        index = np.arange(n_groups)
        bar_width = 0.18
        for i, model_name in enumerate(model_names):
            scores = [results_dict[model_name][trans][metric] for trans in transform_names]
            plt.bar(index + i * bar_width, scores, bar_width, label=model_name)
        plt.xlabel('Geometric Transformation', fontsize=12)
        plt.ylabel(f'Mean {metric.capitalize()} (%)', fontsize=12)
        plt.title(f'Model Robustness to Transformations ({metric.capitalize()})', fontsize=16)
        plt.xticks(index + bar_width * (len(model_names) - 1) / 2, transform_names, rotation=20, ha="right")
        plt.legend()
        plt.grid(axis='y', linestyle='--')

    def run(self):
        try:
            # Load the saved training config first
            self.update_log("--- 1. Loading Training Configuration ---")
            config_path = os.path.join(self.config['output_dir'], "training_config.json")

            if not os.path.exists(config_path):
                error_msg = f"Configuration file not found at {config_path}. Please run training for this output directory first."
                self.update_log(f"ERROR: {error_msg}")
                self.socketio.emit('benchmark_error', {'error': error_msg})
                return

            with open(config_path, 'r') as f:
                training_config = json.load(f)

            keys_to_override = ['unet_features', 'film_hidden_dim', 'patch_hidden_dim', 'img_size', 'data_subset']
            for key in keys_to_override:
                if key in training_config:
                    self.config[key] = training_config[key]
            self.update_log("Successfully loaded and applied training configuration.")

            # Load Data, Maps, and Caches
            self.update_log("--- 2. Loading Data, Maps, and Caches ---")
            class_idx_to_rgb, num_classes = self.create_color_maps(self.config['class_csv'])
            all_image_files = sorted(glob(os.path.join(self.config['processed_image_dir'], '*.png')))
            all_mask_files = sorted(glob(os.path.join(self.config['processed_mask_dir'], '*.png')))
            _, X_val_paths, _, y_val_paths = train_test_split(all_image_files, all_mask_files, test_size=0.2, random_state=42)
            self.update_log(f" Loaded {len(X_val_paths)} validation image paths.")
            feature_caches = {}
            cache_names = ['cm_global', 'fmt_global', 'cm_patch']
            for name in cache_names:
                cache_path = os.path.join(self.config['cache_dir'], f"{name}_cache_{self.config['data_subset']}.pkl")
                with open(cache_path, 'rb') as f:
                    feature_caches[name] = pickle.load(f)
            self.update_log(" All feature caches loaded.")

            # Load Pre-Trained Models
            self.update_log("--- 3. Loading Pre-Trained Models ---")
            cm_global_len = next(iter(feature_caches['cm_global'].values())).shape[0]
            fmt_global_len = next(iter(feature_caches['fmt_global'].values())).shape[0]
            cm_patch_len = next(iter(feature_caches['cm_patch'].values())).shape[-1]
            
            models_to_evaluate = {
                "Baseline": {"model": UNet(num_classes=num_classes, features=self.config['unet_features']), "feature_type": "none", "cache_key": None},
                "FiLM_CM": {"model": UNetWithFiLM(num_classes=num_classes, feature_len=cm_global_len, features=self.config['unet_features'], film_hidden_dim=self.config['film_hidden_dim']), "feature_type": "global", "cache_key": "cm_global"},
                "FiLM_FMT": {"model": UNetWithFiLM(num_classes=num_classes, feature_len=fmt_global_len, features=self.config['unet_features'], film_hidden_dim=self.config['film_hidden_dim']), "feature_type": "global", "cache_key": "fmt_global"},
                "Patch_CM": {"model": UNetWithPatchFeatures(num_classes=num_classes, feature_len=cm_patch_len, features=self.config['unet_features'], patch_hidden_dim=self.config['patch_hidden_dim']), "feature_type": "patch", "cache_key": "cm_patch"},
            }

            for name, config in models_to_evaluate.items():
                model_path = os.path.join(self.config['saved_models_dir'], f"{name}_model.pth")
                if not os.path.exists(model_path):
                    self.update_log(f" ERROR: Model file not found at {model_path}. Please train the models first.")
                    return
                config['model'].load_state_dict(torch.load(model_path, map_location=self.device))
                config['model'].to(self.device)
            self.update_log(" All pre-trained models loaded.")

            # Feature Visualization (t-SNE) 
            self.update_log("--- 4.  Feature Visualization (t-SNE) ---")
            self.plot_and_save(self._tsne_plotter, feature_caches['cm_global'], all_mask_files, "t-SNE of Global Complex Moments", filename="tsne_cm_global.png")
            self.plot_and_save(self._tsne_plotter, feature_caches['fmt_global'], all_mask_files, "t-SNE of Global Fourier-Mellin", filename="tsne_fmt_global.png")

            # Quantitative Evaluation
            self.update_log("--- 5.  Quantitative Evaluation ---")
            img_size = (self.config['img_size'], self.config['img_size'])
            transformations = {
                "Original": T.Compose([T.Resize(img_size)]),
                "Rotate 45°": T.Compose([T.RandomRotation(degrees=(45, 45)), T.Resize(img_size)]),
                "H-Flip": T.Compose([T.RandomHorizontalFlip(p=1.0), T.Resize(img_size)]),
                "Scale & Crop": T.Compose([T.Resize(int(img_size[0] * 1.2)), T.CenterCrop(img_size)]),
                "Affine Mix": T.Compose([T.RandomAffine(degrees=15, translate=(0.1, 0.1)), T.Resize(img_size)])
            }
            robustness_results = {name: {} for name in models_to_evaluate.keys()}
            for trans_name, trans_func in transformations.items():
                self.update_log(f"--- Evaluating with transformation: {trans_name} ---")
                for name, config in models_to_evaluate.items():
                    val_dset = self.EvaluationDataset(X_val_paths, y_val_paths, feature_cache=feature_caches.get(config['cache_key']), feature_type=config['feature_type'], transform=trans_func, img_size=img_size)
                    val_loader = DataLoader(val_dset, batch_size=self.config['batch_size'])
                    acc, iou, dice = self.evaluate_model(val_loader, config['model'], num_classes, config['feature_type'])
                    robustness_results[name][trans_name] = {'accuracy': acc, 'iou': iou, 'dice': dice}
                    self.update_log(f"  - {name}: Acc={acc:.2f}%, IoU={iou:.2f}%, Dice={dice:.2f}%")
            self.socketio.emit('benchmark_result', {'type': 'table', 'data': robustness_results})

            #Visualize Segmentation Outputs
            self.update_log("--- 6.  Visualize Segmentation Outputs ---")
            sample_idx = random.randint(0, len(X_val_paths) - 1)
            sample_img_path = X_val_paths[sample_idx]
            sample_data = {'img_path': sample_img_path, 'mask_path': y_val_paths[sample_idx]}
            for name in cache_names:
                sample_data[name] = feature_caches[name][sample_img_path]
            self.plot_and_save(self._comparison_grid_plotter, sample_data, models_to_evaluate, class_idx_to_rgb, filename="segmentation_comparison.png")
            
            #Plot Final Robustness Charts 
            self.update_log("--- 7. Plot Final Robustness Charts ---")
            self.plot_and_save(self._robustness_plotter, robustness_results, 'accuracy', filename="robustness_accuracy.png")
            self.plot_and_save(self._robustness_plotter, robustness_results, 'iou', filename="robustness_iou.png")

            self.update_log(" Evaluation and visualization complete.")
            self.socketio.emit('benchmark_complete')
            
        except Exception as e:
            self.update_log(f"An error occurred during benchmarking: {e}")
            self.socketio.emit('benchmark_error', {'error': str(e)})