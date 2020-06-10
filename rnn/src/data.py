import random
from os import walk

import librosa

from rnn.src.config import ModelConfig
from rnn.src.preprocess import get_random_wav, to_spectrogram, get_magnitude, spec_to_batch
from rnn.src.config import TrainConfig


class Data:
    def __init__(self, path):
        self.path = path

    def next_wavs(self, size=1):
        wavfiles = []
        for (root, dirs, files) in walk(self.path):
            wavfiles.extend(['{}/{}'.format(root, f) for f in files if f.endswith(".wav")])
        if(size == -1):
            size = len(wavfiles)
        wav = random.sample(wavfiles, size)
        mixed, src1, src2 = get_random_wav(wav, TrainConfig.SECONDS, ModelConfig.SR)
        return mixed, src1, src2

    def prepare_data(self, wav):
        wav_spec = to_spectrogram(wav)
        wav_mag = get_magnitude(wav_spec)
        wav_batch, _ = spec_to_batch(wav_mag)
        return wav_batch

if __name__ == "__main__":
    wav = librosa.load("../dataset/mix/3.wav", sr=8000, mono=False)
    print("end")