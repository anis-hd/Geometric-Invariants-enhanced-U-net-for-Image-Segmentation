# feature_calculators.py

import numpy as np
import cv2

class CMCalc:
    """Calculates Complex Moments."""
    def _centroid(self, X: np.ndarray):
        h, w = X.shape
        Y, Xc = np.mgrid[0:h, 0:w]
        m00 = np.sum(X)
        if m00 == 0: return w / 2, h / 2, 0
        m10 = np.sum(Xc * X)
        m01 = np.sum(Y * X)
        return m10 / m00, m01 / m00, m00

    def cm_rgb(self, I: np.ndarray, N: int = 2):
        Mall = []
        for k in range(I.shape[2]):
            X = I[:, :, k].astype(np.float64)
            h, w = X.shape
            xbar, ybar, m00 = self._centroid(X)
            Y, Xc = np.mgrid[0:h, 0:w]
            Z = (Xc - xbar) + 1j * (Y - ybar)
            M = []
            m00n = m00 if m00 != 0 else 1.0
            for p in range(N + 1):
                for q in range(N + 1):
                    if p + q >= 2:
                        mu = np.sum(X * (Z ** p) * (np.conj(Z) ** q))
                        nrm = m00n ** (((p + q) / 2.0) + 1.0)
                        nu = mu / nrm if nrm != 0 else 0
                        M.append(nu)
            Mall.append(np.abs(np.array(M)))
        return np.concatenate(Mall)


class FMCalc:
    """Calculates Fourier-Mellin Descriptors."""
    def _fm_single(self, x: np.ndarray, n: int = 4096): # Default n to match 64*64
        x = x.astype(np.float32)
        x = cv2.GaussianBlur(x, (5, 5), 0)
        h, w = x.shape
        c = (w / 2, h / 2)
        r_max = np.sqrt((h / 2) ** 2 + (w / 2) ** 2)
        if r_max == 0: return np.zeros(n)
        if r_max < 1: r_max = 1.0
        
        # This check is critical for stability
        M = w / np.log(r_max) if np.log(r_max) > 0 else 0
        if M <= 0: return np.zeros(n)

        xp = cv2.logPolar(x, c, M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        F = cv2.dft(xp, flags=cv2.DFT_COMPLEX_OUTPUT)
        F = np.fft.fftshift(F)
        mag = cv2.magnitude(F[:, :, 0], F[:, :, 1]) + 1e-9
        L = np.log(mag)
        m = L.flatten()
        s = np.std(m)
        if s > 0:
            m = (m - np.mean(m)) / s
        
        if len(m) < n:
            return np.pad(m, (0, n - len(m)), 'constant')
        return m[:n]

    def fm_rgb(self, I: np.ndarray, n: int = 4096):
        M_all = [self._fm_single(I[:, :, k], n) for k in range(I.shape[2])]
        return np.concatenate(M_all)