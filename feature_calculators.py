import numpy as np
import cv2

class InvariantMethods:

    def _compute_central_moments(self, channel):
        h, w = channel.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        m00 = np.sum(channel)
        if m00 == 0: return 0, 0, m00
        x_bar = np.sum(x_coords * channel) / m00
        y_bar = np.sum(y_coords * channel) / m00
        return x_bar, y_bar, m00




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




    def fourier_mellin_descriptor(self, image_rgb, log_polar_size=(64, 64)):

        descriptors = []
        for c in range(image_rgb.shape[2]):
            channel = image_rgb[:, :, c].astype(np.float32)
            f_transform = np.fft.fft2(channel)
            f_transform_shifted = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log1p(np.abs(f_transform_shifted))
            center = (channel.shape[1] / 2, channel.shape[0] / 2)
            max_radius = np.sqrt(((channel.shape[0]/2)**2) + ((channel.shape[1]/2)**2))
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
                patch = image_rgb[i*stride : i*stride+patch_size, j*stride : j*stride+patch_size]
                feature_map[i, j, :] = method_func(patch)
                
        return feature_map