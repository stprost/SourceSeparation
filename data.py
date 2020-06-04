import random
from os import walk
from config import ModelConfig
from preprocess import get_random_wav
from config import TrainConfig


class Data:
    def __init__(self, path):
        self.path = path

    def next_wavs(self, size=1):
        wavfiles = []
        for (root, dirs, files) in walk(self.path):
            wavfiles.extend(['{}/{}'.format(root, f) for f in files if f.endswith(".wav")])
        wav = random.sample(wavfiles, size)
        mixed, src1, src2 = get_random_wav(wav, TrainConfig.SECONDS, ModelConfig.SR)
        return mixed, src1, src2

