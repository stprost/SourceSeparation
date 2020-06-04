from os import walk

import librosa
import re
import numpy as np
from scipy.io.wavfile import write


def write_stereo(path, wav_1, wav_2):
    pad_len = np.abs(wav_1.size - wav_2.size)
    if wav_1.size < wav_2.size:
        wav_1 = np.pad(wav_1, (0, pad_len), 'constant', constant_values=0)
    elif wav_2.size < wav_1.size:
        wav_2 = np.pad(wav_2, (0, pad_len), 'constant', constant_values=0)
    write(path, 22050, np.array([wav_1, wav_2]).transpose())


def generate(num, path_input, path_output):
    wavfiles = []
    for (root, dirs, files) in walk(path_input):
        wavfiles.append(['{}/{}'.format(root, f) for f in files[len(files)-num:] if f.endswith(".wav")])
    wavfiles = wavfiles[1:]
    for i in range(len(wavfiles) - 1):
        for j in range(i + 1, len(wavfiles)):
            for wav_1 in wavfiles[i]:
                for wav_2 in wavfiles[j]:
                    name_1 = re.split("\.", re.split("/", wav_1)[3])[0]
                    name_2 = re.split("\.", re.split("/", wav_2)[3])[0]
                    name = '{}__{}.wav'.format(name_1, name_2)
                    write_stereo("{}/{}".format(path_output, name), np.array(librosa.load(wav_1)[0]),
                                 np.array(librosa.load(wav_2)[0]))


def generate_m_f(num, path_male, path_female, path_output):
    wavfiles_male = []
    for (root, dirs, files) in walk(path_male):
        wavfiles_male.extend(['{}/{}'.format(root, f) for f in files[len(files)-num:] if f.endswith(".wav")])
    wavfiles_male = wavfiles_male[1:]
    wavfiles_female = []
    for (root, dirs, files) in walk(path_female):
        wavfiles_female.extend(['{}/{}'.format(root, f) for f in files[len(files)-num:] if f.endswith(".wav")])
    wavfiles_female = wavfiles_female[1:]
    for wav_1 in wavfiles_male:
        for wav_2 in wavfiles_female:
            name_1 = re.split("\.", re.split("/", wav_1)[3])[0]
            name_2 = re.split("\.", re.split("/", wav_2)[3])[0]
            name = '{}__{}.wav'.format(name_1, name_2)
            write_stereo("{}/{}".format(path_output, name), np.array(librosa.load(wav_1)[0]),
                         np.array(librosa.load(wav_2)[0]))


if __name__ == "__main__":
    generate(1, "48k/male", "dataset/test/m_m")
    generate(1, "48k/female", "dataset/test/f_f")
    generate_m_f(1, "48k/male", "48k/female", "dataset/test/m_f")
