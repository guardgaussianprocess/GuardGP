import math
from collections import deque
import numpy as np
SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0*math.pi)


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def surprise_from_gaussian(y, mu, var, two_sided: bool = True,
                           sigma_floor: float | None = None,
                           add_noise: float | None = None,
                           inflate: float = 1.0) -> float:
   
    sigma = max(float(var), 0.0)**0.5
    if add_noise is not None:
        sigma = (sigma*sigma + add_noise*add_noise)**0.5
    if sigma_floor is not None:
        sigma = max(sigma, sigma_floor)
    sigma *= float(inflate)

    z = abs((float(y) - float(mu)) / max(sigma, 1e-12))

    if two_sided:
        u = math.erfc(z / SQRT2)        
    else:
        u = 0.5 * math.erfc(z / SQRT2) 

    if u > 0.0:
        return -math.log(u)

    logu = -0.5*z*z - math.log(max(z*SQRT2PI, 1e-12))
    return -logu


class SpotEVTExp:
  
    def __init__(self,
                 q_hi=0.005, q_lo=0.02,
                 init_quantile=0.90,
                 min_peaks=10,
                 max_peaks=50):
        assert 0 < q_hi < q_lo < 1
        self.q_hi = q_hi
        self.q_lo = q_lo
        self.init_quantile = init_quantile
        self.min_peaks = min_peaks
        self.max_peaks = max_peaks

        self.tE = None
        self.peaks = deque()   
        self.sigma_hat = 1e-6  
        self.n_seen = 0        

        self.z_lo = float('inf')
        self.z_hi = float('inf')

    def _fit_sigma(self):
        if len(self.peaks) == 0:
            self.sigma_hat = 1e-6
        else:
            self.sigma_hat = max(sum(self.peaks) / len(self.peaks), 1e-9)

    def _update_thresholds(self):
        Nt = len(self.peaks)
        if Nt <= 0 or self.n_seen <= 0:
            self.z_lo = self.z_hi = float('inf')
            return
        ratio = lambda q: Nt / (q * self.n_seen + 1e-12)
        
        self.z_lo = self.tE + self.sigma_hat * max(math.log(max(ratio(self.q_lo), 1e-12)), 0.0)
        self.z_hi = self.tE + self.sigma_hat * max(math.log(max(ratio(self.q_hi), 1e-12)), 0.0)

    
    def initialize_from_clean_E(
        self,
        E_clean: list,
        fallback_E_all: list | None = None,
        use_fallback: bool = False,    
        E_clip_init: float = 20.0,      
        trim_top: float = 0.10,         
        sigma_floor: float = 1e-9       
    ):
        

        arr = np.asarray(E_clean, dtype=float)

        if use_fallback and arr.size < max(30, self.min_peaks + 10) and fallback_E_all is not None:
            all_arr = np.asarray(fallback_E_all, dtype=float)
            k = max(3, int(0.10 * all_arr.size))
            all_sorted = np.sort(all_arr)
            arr = all_sorted[:-k] if all_sorted.size > k else all_sorted

        
        if E_clip_init is not None:
            arr = np.minimum(arr, float(E_clip_init))

        self.n_seen = int(arr.size)

       
        p = self.init_quantile
        peaks = []
        while p >= 0.80:
            self.tE = float(np.quantile(arr, p))
            raw_peaks = np.asarray([e - self.tE for e in arr if e > self.tE], dtype=float)
            if raw_peaks.size >= self.min_peaks:
                
                if trim_top > 0 and raw_peaks.size > 4:
                    q_cut = float(np.quantile(raw_peaks, 1.0 - trim_top))
                    raw_peaks = np.minimum(raw_peaks, q_cut)
                peaks = raw_peaks
                break
            p -= 0.02

       
        if len(peaks) == 0:
            self.tE = float(np.quantile(arr, 0.80))
            raw_peaks = np.asarray([e - self.tE for e in arr if e > self.tE], dtype=float)
            if trim_top > 0 and raw_peaks.size > 4:
                q_cut = float(np.quantile(raw_peaks, 1.0 - trim_top))
                raw_peaks = np.minimum(raw_peaks, q_cut)
            peaks = raw_peaks

       
        self.peaks.clear()
        for y in peaks[-self.max_peaks:]:
            self.peaks.append(float(y))

        if len(self.peaks) == 0:
            self.sigma_hat = 1e-6
        else:
            med = float(np.median(self.peaks))
            self.sigma_hat = max(med / math.log(2.0), sigma_floor)

        self._update_thresholds()


  
    def step(self, E_t: float):
        

        if E_t > 2*self.z_hi:
            label = 'attack'
            
            info = {
            "tE": self.tE, "z_lo": self.z_lo, "z_hi": self.z_hi,
            "sigma": self.sigma_hat, "n_seen": self.n_seen,
            "n_peaks": len(self.peaks)
        }
            return label, info

        self.n_seen += 1
        if E_t > self.z_lo:
            label = 'uncertain'
            self.peaks.append(E_t - self.tE)
            if len(self.peaks) > self.max_peaks:
                self.peaks.popleft()
            self._fit_sigma()
            self._update_thresholds()
        
        else:
            label = 'clean'
            self._update_thresholds()

        info = {
            "tE": self.tE, "z_lo": self.z_lo, "z_hi": self.z_hi,
            "sigma": self.sigma_hat, "n_seen": self.n_seen,
            "n_peaks": len(self.peaks)
        }
        return label, info
