import io
import PIL
import h5py
import numpy as np
import mne.time_frequency.tfr
import matplotlib.pyplot as plt

class CWT:

    def __init__(self, x: np.ndarray, 
                 sfreq: int = 250, 
                 freqs: np.ndarray = np.linspace(1, 30, 200)):
        
        self.time = np.arange(x.shape[2]) * 1 / sfreq
        self.freqs = freqs

        # https://mne.tools/dev/generated/mne.time_frequency.tfr_array_morlet.html
        
        cycles = 199 * (np.pi / 5) * (freqs/sfreq)
        wavelets = mne.time_frequency.tfr.morlet(sfreq=sfreq,
                                                 freqs=freqs,
                                                 n_cycles=cycles)

        # Takes a long time!!
        self.coefs = np.stack([mne.time_frequency.tfr.cwt(x[n], wavelets) for n in range(x.shape[0])])
        self.coefs_abs = np.abs(self.coefs)
    
    def plot_n(self, n: int, c: int = 0):
        """
        Display CWT plot as Scaleogram
        """
        pcm = plt.pcolormesh(self.time, self.freqs, self.coefs_abs[n][c])
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Continuous Wavelet Transform (Scaleogram)")
        plt.colorbar(pcm)
        plt.show()

    def get_png(self, n: int, c :int = 0, out_w = 640, out_h = 480):
        with io.BytesIO() as buf:
            plt.pcolormesh(self.time, self.freqs, self.coefs_abs[n][c])
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            return PIL.Image.open(buf).resize((out_w, out_h))
        
    def fill_db(self, db: h5py.Dataset, sind: int = 0):
        if self.coefs_abs.shape[1] == 1:
            for n in range(self.coefs_abs.shape[0]):
                db[n+sind] = np.asarray(self.get_png(n, out_w = 299, out_h = 299))[:,:,:3] # For inception
        else:
            for n in range(self.coefs_abs.shape[0]):
                for c in range(self.coefs_abs.shape[1]):
                    db[n+sind, c] = np.asarray(self.get_png(n, c, 299, 299))[:,:,:3] # For inception