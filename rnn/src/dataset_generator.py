from os import walk

import librosa
import re
import numpy as np
from scipy.io.wavfile import write
from rnn.src.data import Data


def write_stereo(path, wav_1, wav_2):
    pad_len = np.abs(wav_1.size - wav_2.size)
    if wav_1.size < wav_2.size:
        wav_1 = np.pad(wav_1, (0, pad_len), 'constant', constant_values=0)
    elif wav_2.size < wav_1.size:
        wav_2 = np.pad(wav_2, (0, pad_len), 'constant', constant_values=0)
    write(path, 22050, np.array([wav_1, wav_2]).transpose())


def write_mono(path, wav_1, wav_2):
    pad_len = np.abs(wav_1.size - wav_2.size)
    if wav_1.size < wav_2.size:
        wav_1 = np.pad(wav_1, (0, pad_len), 'constant', constant_values=0)
    elif wav_2.size < wav_1.size:
        wav_2 = np.pad(wav_2, (0, pad_len), 'constant', constant_values=0)
    wav = np.concatenate((wav_1, wav_2))
    wav = np.expand_dims(wav, axis=0)
    wav = np.reshape(wav, (2, -1))
    wav_mono = librosa.to_mono(wav)
    write(path, 22050, wav_mono)


def generate(num, path_input, path_output):
    wavfiles = []
    for (root, dirs, files) in walk(path_input):
        wavfiles.append(['{}/{}'.format(root, f) for f in files[len(files) - num:] if f.endswith(".wav")])
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


def generate_m_f(start, end, path_male, path_female, path_output):
    wavfiles_male = []
    for (root, dirs, files) in walk(path_male):
        wavfiles_male.extend(['{}/{}'.format(root, f) for f in files[start:end] if f.endswith(".wav")])
    wavfiles_male = wavfiles_male[1:]
    wavfiles_female = []
    for (root, dirs, files) in walk(path_female):
        wavfiles_female.extend(['{}/{}'.format(root, f) for f in files[start:end] if f.endswith(".wav")])
    wavfiles_female = wavfiles_female[1:]
    f = open("{}/file_name_list.txt".format(path_output), 'w')
    for i in range(len(wavfiles_male)):
        for j in range(len(wavfiles_female)):
            name_1 = re.split("\.", re.split("\\\\|/", wavfiles_male[i])[3])[0]
            name_2 = re.split("\.", re.split("\\\\|/", wavfiles_female[j])[3])[0]
            if re.search("[0-9]{2}_[0-9]{2}", name_1).group(0) == re.search("[0-9]{2}_[0-9]{2}", name_2).group(0):
                continue
            name = '{}__{}'.format(name_1, name_2)
            f.write(name + '\n')
            write_mono("{}/mix/{}.wav".format(path_output, name),
                       np.array(librosa.load(wavfiles_male[i])[0]), np.array(librosa.load(wavfiles_female[j])[0]))
            write("{}/s1/{}.wav".format(path_output, name), 22050, np.array(librosa.load(wavfiles_male[i])[0]))
            write("{}/s2/{}.wav".format(path_output, name), 22050, np.array(librosa.load(wavfiles_female[j])[0]))
            # write_stereo("{}/{}".format(path_output, name), np.array(librosa.load(wav_1)[0]),
            #              np.array(librosa.load(wav_2)[0]))
    f.close()


def save_array():
    data_train = Data("dataset/train/m_f")
    mixed_wav, src1_wav, src2_wav = data_train.next_wavs(-1)
    mixed_batch, src1_batch, src2_batch = data_train.prepare_data(mixed_wav), data_train.prepare_data(
        src1_wav), data_train.prepare_data(src2_wav)
    np.savetxt("dataset/train/mixed_batch.csv",
               mixed_batch.reshape((mixed_batch.shape[0] * mixed_batch.shape[1], mixed_batch.shape[2])), delimiter=',')
    np.savetxt("dataset/train/src1_batch.csv",
               src1_batch.reshape((src1_batch.shape[0] * src1_batch.shape[1], src1_batch.shape[2])), delimiter=',')
    np.savetxt("dataset/train/src2_batch.csv",
               src2_batch.reshape((src2_batch.shape[0] * src2_batch.shape[1], src2_batch.shape[2])), delimiter=',')


if __name__ == "__main__":
    # data = Data("dataset/train/m_f")
    # mixed_wav, src1_wav, src2_wav = data.next_wavs(-1)
    # for i in range(mixed_wav.shape[0]):
    #     write("dataset/mix/{}.wav".format(i), 16000, mixed_wav[i])
    # generate(1, "48k/male", "dataset/test/m_m")
    # generate(1, "48k/female", "dataset/test/f_f")
    generate_m_f(0, 5, "../48k/male", "48k/female", "ds/tr")
    generate_m_f(5, 7, "../48k/male", "48k/female", "ds/cv")
    generate_m_f(7, 9, "../48k/male", "48k/female", "ds/tt")
