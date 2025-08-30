
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from PIL import Image
import eventlet
import base64
from io import BytesIO


class InvariantCalculator:
    """
    Manages the calculation and visualization of geometric invariants for an image.
    """
    def __init__(self, config, socketio):
        self.config = config
        self.socketio = socketio
        self.update_log("Invariant Calculator initialized.")

    def update_log(self, message):
        """Sends a log message to the frontend and yields control to the server."""
        print(message)
        self.socketio.emit('invariant_log', {'data': message})
        eventlet.sleep(0.01)

    ############# FOURIER-MELLIN CALCULATION

    class FMCalc:
        """Calculates Fourier-Mellin moments, which are invariant to rotation, scale, and translation."""

        def _fm_single(self, x: np.ndarray, n: int = 500):
            # Pre-process the image channel
            x = x.astype(np.float32); x = cv2.GaussianBlur(x, (5, 5), 0)
            h, w = x.shape; c = (w / 2, h / 2)
            
            # Log-Polar Transform:
            # A rotation and scaling in the Cartesian (x,y) coordinate system
            # becomes a simple translation in the log-polar (log(r), theta) coordinate system
            r_max = np.sqrt((h/2)**2 + (w/2)**2)
            if r_max == 0: return np.zeros(n)
            if r_max < 1: r_max = 1.0
            M = w / np.log(r_max) 
            xp = cv2.logPolar(x, c, M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
            
            # Fourier Transform's Shift:
            # If an image is shifted, its fourier magnitude spectrum remains the same
            F = cv2.dft(xp, flags=cv2.DFT_COMPLEX_OUTPUT); F = np.fft.fftshift(F)
            
            # By taking the Fourier magnitude of the log-polar image, we create a descriptor that is
            # invariant to shifts in the log-polar domain, which correspond to rotation and scaling
            # in the original cartesian domain
            mag = cv2.magnitude(F[:, :, 0], F[:, :, 1]) + 1e-9
            
            # Use log of magnitude to compress the range, and then normalize
            # to make the descriptor robust to contrast changes
            L = np.log(mag); m = L.flatten(); s = np.std(m)
            if s > 0: m = (m - np.mean(m)) / s

            if len(m) < n: return np.pad(m, (0, n - len(m)), 'constant')
            return m[:n]

        def fm_rgb(self, I: np.ndarray, n: int = 500):
            M_all = [self._fm_single(I[:, :, k], n) for k in range(I.shape[2])]
            return np.concatenate(M_all)
        








    ################   COMPLEX MOMENT CALCULATION   #########################

    class CMCalc:


        
        # Calculate centroid of the image
        def _centroid(self, X: np.ndarray):
            h, w = X.shape; Y, Xc = np.mgrid[0:h, 0:w]; m00 = np.sum(X) # m00 is the total mass
            if m00 == 0: return w/2, h/2, 0
            # m10 and m01 are weighted sums of coordinates
            m10 = np.sum(Xc * X); m01 = np.sum(Y * X)
            return m10/m00, m01/m00, m00

        def cm_rgb(self, I: np.ndarray, N: int = 2):
            Mall = []
            for k in range(I.shape[2]):
                X = I[:, :, k].astype(np.float64); h, w = X.shape
                
                # Central Moments:
                # Calculate moments relative to the image's centroid
                xbar, ybar, m00 = self._centroid(X)
                
                # Represent coordinates as complex numbers centered at the centroid
                Y, Xc = np.mgrid[0:h, 0:w]; Z = (Xc - xbar) + 1j * (Y - ybar)
                
                M = []; m00n = m00 if m00 != 0 else 1.0
                
                for p in range(N + 1):
                    for q in range(N + 1):
                        if p + q >= 2:
                            mu = np.sum(X * (Z**p) * (np.conj(Z)**q))
                            nrm = m00n ** (((p + q) / 2.0) + 1.0)
                            nu = mu / nrm if nrm != 0 else 0
                            M.append(nu)
                
                # The magnitude of the complex moment is rotation invariant
                Mall.append(np.abs(np.array(M)))
            
            return np.concatenate(Mall)

    ########### visualization and transformation functions

    def _plot_moments(self, moments, title, method):
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        n = len(moments) // 3
        x = np.arange(len(moments)); colors = ['r']*n + ['g']*n + ['b']*n
        ax.bar(x, moments, color=colors, width=1.0)
        ax.set_title(title); ax.set_xlabel("Index (R,G,B)")
        if method == 'cm':
            ax.set_ylabel("Magnitude (log scale)"); ax.set_yscale('symlog')
        else:
            ax.set_ylabel("Normalized Magnitude")
            M = np.max(np.abs(moments)) if len(moments) > 0 else 1
            ax.set_ylim(-1.5*M, 1.5*M)
        
        buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

    def _apply_transform(self, image, transform_name):
        h, w = image.shape[:2]
        if transform_name == "Rotate 45°":
            c = (w // 2, h // 2); R = cv2.getRotationMatrix2D(c, 45, 1.0)
            return cv2.warpAffine(image, R, (w, h))
        elif transform_name == "Flip Horizontal":
            return cv2.flip(image, 1)
        elif transform_name == "Scale 0.8x":
            new_w = max(1, int(w * 0.8)); new_h = max(1, int(h * 0.8))
            return cv2.resize(image, (new_w, new_h))
        return image

    ####################    MAIN        #############
    def run(self):
        try:
            image_path = self.config['image_path']
            method = self.config['method']
            n_moments = 1500
            
            self.update_log(f"Loading image: {os.path.basename(image_path)}")
            I_bgr = cv2.imread(image_path)
            if I_bgr is None: raise FileNotFoundError(f"Could not read image file at {image_path}")
            I_rgb = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2RGB)

            if method == 'cm':
                calculator = self.CMCalc(); calc_func = lambda img: calculator.cm_rgb(img)
            else: # fm
                calculator = self.FMCalc(); calc_func = lambda img: calculator.fm_rgb(img, n=n_moments)

            self.update_log(f"Calculating moments for Original Image using {method.upper()}...")
            original_moments = calc_func(I_rgb)
            original_plot_b64 = self._plot_moments(original_moments, f"Original {method.upper()}", method)

            results = [{'name': 'Original', 'image_path': image_path, 'plot_b64': original_plot_b64, 'similarity': 1.0}]

            transforms = ["Rotate 45°", "Flip Horizontal", "Scale 0.8x"]
            for t_name in transforms:
                self.update_log(f"Applying transform: {t_name}")
                transformed_img_rgb = self._apply_transform(I_rgb.copy(), t_name)
                
                self.update_log(f"Calculating moments for {t_name} using {method.upper()}...")
                transformed_moments = calc_func(transformed_img_rgb)
                
                # Cosine similarity

                norm1 = np.linalg.norm(original_moments); norm2 = np.linalg.norm(transformed_moments)
                similarity = 0.0
                if norm1 > 0 and norm2 > 0:
                    len1, len2 = len(original_moments), len(transformed_moments)
                    moments1_padded, moments2_padded = original_moments, transformed_moments
                    if len1 != len2:
                        if len1 > len2: moments2_padded = np.pad(transformed_moments, (0, len1 - len2))
                        else: moments1_padded = np.pad(original_moments, (0, len2 - len1))
                    
                    similarity = float(np.dot(moments1_padded, moments2_padded) / (np.linalg.norm(moments1_padded) * np.linalg.norm(moments2_padded)))
                
                self.update_log(f"  - Cosine Similarity: {similarity:.4f}")

                transformed_pil = Image.fromarray(transformed_img_rgb)
                temp_filename = f"temp_{t_name.replace(' ', '_').lower()}.png"
                temp_path = os.path.join('static', 'uploads', temp_filename)
                transformed_pil.save(temp_path)
                
                plot_b64 = self._plot_moments(transformed_moments, f"{t_name} {method.upper()}", method)
                
                results.append({'name': t_name, 'image_path': temp_path, 'plot_b64': plot_b64, 'similarity': similarity})

            self.socketio.emit('invariant_result', {'results': results})
            self.update_log(" Invariant calculation complete.")
        
        except Exception as e:
            self.update_log(f" An error occurred during invariant calculation: {e}")
            self.socketio.emit('invariant_error', {'error': str(e)})