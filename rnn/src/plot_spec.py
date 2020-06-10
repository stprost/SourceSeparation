import matplotlib.pyplot as plt
import numpy as np
from librosa import amplitude_to_db, stft
from librosa.display import specshow


def plot(spec):
    specshow(amplitude_to_db(spec, ref=np.max), y_axis='log', x_axis='time')
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
