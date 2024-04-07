import io
import PIL
import h5py
import numpy as np
import pywt
import matplotlib.pyplot as plt
import matplotlib as mpl

class CWT_PYWT:

    def __init__(self, x: np.ndarray, 
                 sfreq: int = 250, 
                 wavelet: str = 'morl',
                 freqs: np.ndarray = np.linspace(1, 30, 100),
                 vmin: float = 1,
                 vmax: float = -1):
        
        widths = pywt.frequency2scale(wavelet, freqs / sfreq)
        self.time = np.arange(x.shape[1]) * 1 / sfreq
        self.coefs = np.zeros((x.shape[0], len(widths), x.shape[1]))
        self.freqs = np.zeros((x.shape[0], len(widths)))

        if vmax != -1:
            self.norm = mpl.colors.LogNorm(vmin = vmin, vmax=vmax)
        else:
            self.norm = None

        # Takes a long time!!
        for n in range(x.shape[0]):
            self.coefs[n], self.freqs[n] = pywt.cwt(x[n], widths, wavelet=wavelet, sampling_period=1/sfreq)
        self.coefs_abs = np.abs(self.coefs)
    
    def plot_n(self, n: int, c: int = 0):
        """
        Display CWT plot as Scaleogram
        """
        
        pcm = plt.pcolormesh(self.time, self.freqs[n], self.coefs_abs[n], norm=self.norm, rasterized=True)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Continuous Wavelet Transform (Scaleogram)")
        plt.colorbar(pcm)
        plt.show()

    def get_png(self, n: int, out_w = 640, out_h = 480):
        
        with io.BytesIO() as buf:
            plt.pcolormesh(self.time, self.freqs[n], self.coefs_abs[n], norm=self.norm, rasterized=True)
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            return PIL.Image.open(buf).resize((out_w, out_h))
        
    def fill_db(self, db: h5py.Dataset, sind: int = 0):
        for n in range(self.coefs_abs.shape[0]):
            db[n+sind] = np.asarray(self.get_png(n, out_w = 299, out_h = 299))[:,:,:3] # For inception